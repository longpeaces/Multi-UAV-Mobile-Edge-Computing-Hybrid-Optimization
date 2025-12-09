from environment.user_equipments import UE
from environment.uavs import UAV
import config
import numpy as np


class Env:
    def __init__(self) -> None:
        self._mbs_pos: np.ndarray = config.MBS_POS
        UE.initialize_ue_class()
        self._ues: list[UE] = [UE(i) for i in range(config.NUM_UES)]
        self._uavs: list[UAV] = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step: int = 0
        self._cs_storage: np.ndarray = np.full(config.NUM_CS, config.CS_STORAGE_CAPACITY / 2.0)
        self._grid_energy_cost: float = 0.0
        self._market_energy_cost: float = 0.0

    @property
    def uavs(self) -> list[UAV]:
        return self._uavs

    @property
    def ues(self) -> list[UE]:
        return self._ues

    def reset(self) -> list[np.ndarray]:
        """Resets the environment to an initial state and returns the initial observations."""
        self._ues = [UE(i) for i in range(config.NUM_UES)]
        self._uavs = [UAV(i) for i in range(config.NUM_UAVS)]
        self._time_step = 0
        self._cs_storage = np.full(config.NUM_CS, config.CS_STORAGE_CAPACITY / 2.0)
        self._grid_energy_cost = 0.0
        self._market_energy_cost = 0.0
        return self._get_obs()

    def step(self, actions: np.ndarray, visualize: bool = False) -> tuple[list[np.ndarray], list[float], tuple[float, float, float, float]]:
        """Execute one time step of the simulation with MEC/energy coordination decisions."""
        del visualize  # trajectory visualization is no longer used
        self._time_step += 1

        current_hour: int = (self._time_step // config.SLOTS_PER_HOUR) % config.HOURS_PER_DAY
        self._inject_renewables()

        for uav, action in zip(self._uavs, actions):
            uav.set_decisions(action, current_hour)

        for uav in self._uavs:
            uav.process_requests()

        for ue in self._ues:
            ue.update_service_coverage(self._time_step)

        for uav in self._uavs:
            uav.update_ema_and_cache()
            uav.update_energy_consumption()

        grid_cost, market_cost = self._apply_energy_trades(current_hour)
        rewards, metrics = self._get_rewards_and_metrics(grid_cost, market_cost)

        if self._time_step % config.T_CACHE_UPDATE_INTERVAL == 0:
            for uav in self._uavs:
                uav.gdsf_cache_update()

        # For next time step
        for ue in self._ues:
            ue.update_position()

        for uav in self._uavs:
            uav.reset_for_next_step()

        next_obs: list[np.ndarray] = self._get_obs(current_hour)
        return next_obs, rewards, metrics

    def _get_obs(self, current_hour: int | None = None) -> list[np.ndarray]:
        """Gets the local observation for each UAV agent."""
        if current_hour is None:
            current_hour = (self._time_step // config.SLOTS_PER_HOUR) % config.HOURS_PER_DAY
        # For new time step
        for ue in self._ues:
            ue.generate_request()
        self._associate_ues_to_uavs()
        for uav in self._uavs:
            uav.set_current_requested_files()
            uav.set_neighbors(self._uavs)
        for uav in self._uavs:
            uav.select_collaborator()
        for uav in self._uavs:
            uav.set_freq_counts()

        all_obs: list[np.ndarray] = []
        for uav in self._uavs:
            # Part 1: Own state (position and cache status)
            own_pos: np.ndarray = uav.pos[:2] / np.array([config.AREA_WIDTH, config.AREA_HEIGHT])
            own_cache: np.ndarray = uav.cache.astype(np.float32)
            cs_price = config.MARKET_PRICES[current_hour]
            norm_battery: float = uav.battery_level / config.UAV_BATTERY_CAPACITY
            norm_hour: float = current_hour / float(config.HOURS_PER_DAY)
            own_state: np.ndarray = np.concatenate([own_pos, np.array([norm_battery, cs_price, norm_hour]), own_cache])

            # Part 2: Neighbors state (positions and cache status)
            neighbor_states: np.ndarray = np.zeros((config.MAX_UAV_NEIGHBORS, 2 + config.NUM_FILES))
            neighbors: list[UAV] = sorted(uav.neighbors, key=lambda n: float(np.linalg.norm(uav.pos - n.pos)))[: config.MAX_UAV_NEIGHBORS]
            for i, neighbor in enumerate(neighbors):
                relative_pos: np.ndarray = (neighbor.pos[:2] - uav.pos[:2]) / config.UAV_SENSING_RANGE
                neighbor_cache: np.ndarray = neighbor.cache.astype(np.float32)
                neighbor_states[i, :] = np.concatenate([relative_pos, neighbor_cache])

            # Part 3: State of associated UEs
            ue_states: np.ndarray = np.zeros((config.MAX_ASSOCIATED_UES, 2 + 3))
            ues: list[UE] = sorted(uav.current_covered_ues, key=lambda u: float(np.linalg.norm(uav.pos[:2] - u.pos[:2])))[: config.MAX_ASSOCIATED_UES]
            for i, ue in enumerate(ues):
                delta_pos: np.ndarray = (ue.pos[:2] - uav.pos[:2]) / config.AREA_WIDTH
                req_type, req_size, req_id = ue.current_request
                norm_id: float = float(req_id) / float(config.NUM_FILES)
                norm_size: float = float(req_size) / float(config.MAX_INPUT_SIZE)
                request_info: np.ndarray = np.array([req_type, norm_size, norm_id], dtype=np.float32)
                ue_states[i, :] = np.concatenate([delta_pos, request_info])

            # Part 4: Combine all parts into a single, flat observation vector
            obs: np.ndarray = np.concatenate([own_state, neighbor_states.flatten(), ue_states.flatten()])
            all_obs.append(obs)

        return all_obs

    def _inject_renewables(self) -> None:
        """Update charging platform storage with renewable generation."""
        renewable_gain = np.random.normal(loc=config.CS_RENEWABLE_MEAN, scale=config.CS_RENEWABLE_MEAN * 0.1, size=config.NUM_CS)
        self._cs_storage = np.clip(self._cs_storage + renewable_gain, 0.0, config.CS_STORAGE_CAPACITY)

    def _apply_energy_trades(self, current_hour: int) -> tuple[float, float]:
        """Handle UAV-CS energy trades and compute costs."""
        grid_cost = 0.0
        market_cost = 0.0
        grid_price = config.GRID_PRICES[current_hour]
        market_price = config.MARKET_PRICES[current_hour]

        for uav in self._uavs:
            cs_idx = uav.id % config.NUM_CS
            energy_needed = uav.energy
            available_battery = uav.battery_level

            # Attempt to satisfy demand from battery first
            from_battery = min(available_battery, energy_needed)
            uav.battery_level -= from_battery
            remaining_demand = energy_needed - from_battery

            # Energy trade decision: positive means buy from CS, negative sell surplus
            trade = float(np.clip(uav.energy_trade, -1.0, 1.0))

            if remaining_demand > 0:
                # Buy from CS according to trade share
                buy_from_cs = min(remaining_demand * max(trade, 0.0), config.CS_MAX_TRANSFER, self._cs_storage[cs_idx])
                self._cs_storage[cs_idx] -= buy_from_cs
                market_cost += (buy_from_cs / 3600000.0) * market_price
                remaining_demand -= buy_from_cs

                if remaining_demand > 0:
                    # Fall back to grid for the rest
                    grid_cost += (remaining_demand / 3600000.0) * grid_price
            else:
                surplus = -remaining_demand
                retain_buffer = config.UAV_BATTERY_CAPACITY * config.BASIS_ENERGY_BUFFER
                sellable = max(0.0, uav.battery_level - retain_buffer)
                sell_amount = min(surplus * max(-trade, 0.0), sellable, config.CS_MAX_TRANSFER)
                uav.battery_level -= sell_amount
                self._cs_storage[cs_idx] = min(config.CS_STORAGE_CAPACITY, self._cs_storage[cs_idx] + sell_amount)
                market_cost -= (sell_amount / 3600000.0) * market_price

            # Recharge battery opportunistically from CS if space is available
            recharge = min(config.CS_MAX_TRANSFER, self._cs_storage[cs_idx], config.UAV_BATTERY_CAPACITY - uav.battery_level)
            self._cs_storage[cs_idx] -= recharge
            uav.battery_level += recharge

        self._grid_energy_cost = grid_cost
        self._market_energy_cost = market_cost
        return grid_cost, market_cost

    def _associate_ues_to_uavs(self) -> None:
        """Assigns each UE to at most one UAV, resolving overlaps by choosing the closest UAV."""
        for ue in self._ues:
            covering_uavs: list[tuple[UAV, float]] = []
            for uav in self._uavs:
                distance: float = float(np.linalg.norm(uav.pos[:2] - ue.pos[:2]))
                if distance <= config.UAV_COVERAGE_RADIUS:
                    covering_uavs.append((uav, distance))

            if not covering_uavs:
                continue
            best_uav, _ = min(covering_uavs, key=lambda x: x[1])
            best_uav.current_covered_ues.append(ue)
            ue.assigned = True

    def _get_rewards_and_metrics(self, grid_cost: float, market_cost: float) -> tuple[list[float], tuple[float, float, float, float]]:
        """Returns the reward and other metrics."""
        total_latency: float = sum(ue.latency_current_request if ue.assigned else config.NON_SERVED_LATENCY_PENALTY for ue in self._ues)
        total_energy: float = sum(uav.energy for uav in self._uavs)
        sc_metrics: np.ndarray = np.array([ue.service_coverage for ue in self._ues])
        jfi: float = 0.0
        if sc_metrics.size > 0 and np.sum(sc_metrics**2) > 0:
            jfi = (np.sum(sc_metrics) ** 2) / (sc_metrics.size * np.sum(sc_metrics**2))

        energy_cost = grid_cost + market_cost

        # Normalized, target-aware reward components keep well-performing policies positive
        fairness_term: float = np.log((jfi + config.EPSILON) / max(config.FAIRNESS_TARGET, config.EPSILON))
        latency_term: float = np.log(max(config.LATENCY_TARGET, config.EPSILON) / (total_latency + config.EPSILON))
        energy_term: float = np.log(max(config.ENERGY_COST_TARGET, config.EPSILON) / (energy_cost + config.EPSILON))

        reward: float = config.REWARD_OFFSET
        reward += config.ALPHA_3 * fairness_term
        reward += config.ALPHA_1 * latency_term
        reward += config.ALPHA_2 * energy_term
        rewards: list[float] = [reward * config.REWARD_SCALING_FACTOR for _ in range(config.NUM_UAVS)]
        return rewards, (total_latency, total_energy, jfi, energy_cost)
