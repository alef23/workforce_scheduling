from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class DemandNoiseResult:
    """
    Resultado de la generación de demanda inicial a partir de una matriz de cobertura.
    """

    initial_demand: np.ndarray
    discount_matrix: np.ndarray
    demand_propensity: np.ndarray

    daily_component: np.ndarray
    hourly_component: np.ndarray
    local_noise: np.ndarray
    discount_weights: np.ndarray

    daily_peak_contributions: np.ndarray
    hourly_peak_contributions: np.ndarray

    k_effective: float
    discount_total: int

    daily_peaks: list[dict[str, Any]]
    hourly_peaks: list[dict[str, Any]]


class DemandNoiseGenerator:
    """
    Generador de demanda inicial a partir de una matriz de cobertura factible.

    Entrada principal:
        coverage_matrix = C

    Salida principal:
        initial_demand = D0 = C - R

    Donde:
        C  = matriz de cobertura factible.
        R  = matriz de descuento aleatorio.
        D0 = demanda inicial simulada.

    Garantías:
        D0[h, d] >= 0
        0 <= R[h, d] <= C[h, d]
    """

    def __init__(
        self,
        k: float,
        max_daily_peaks: int = 4,
        max_hourly_peaks: int = 2,
        sigma_lambda: float = 0.50,
        sigma_alpha: float = 0.50,
        sigma_u: float = 0.15,
        epsilon: float = 1e-9,
        chi_square_c: float = 4.0,
        q_baseline: float = 0.3,
        capacity_gamma: float = 0.5,
        min_capacity_factor: float = 0.30,
        seed: Optional[int] = None,
    ) -> None:
        self.k = k
        self.max_daily_peaks = max_daily_peaks
        self.max_hourly_peaks = max_hourly_peaks

        self.sigma_lambda = sigma_lambda
        self.sigma_alpha = sigma_alpha
        self.sigma_u = sigma_u

        self.epsilon = epsilon
        self.chi_square_c = chi_square_c
        self.q_baseline = q_baseline
        self.capacity_gamma = capacity_gamma
        self.min_capacity_factor = min_capacity_factor
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        self._validate_init_params()

        # Sigma corregido:
        # tanto para días como para horas, se elige en {0.5, 1.0, ..., 3.0}.
        self.sigma_choices = self._build_sigma_choices(
            min_value=0.5,
            max_value=3.0,
            step=0.5,
        )

    # ============================================================
    # Método principal
    # ============================================================

    def generate(self, coverage_matrix: np.ndarray) -> DemandNoiseResult:
        """
        Genera una demanda inicial a partir de una matriz de cobertura factible.
        """

        C = self._validate_coverage_matrix(coverage_matrix)

        k_effective = self._sample_k_effective()

        discount_total = self._compute_discount_total(
            coverage_matrix=C,
            k_effective=k_effective,
        )

        (
            daily_component,
            daily_peaks,
            daily_peak_contributions,
        ) = self._generate_daily_component()

        (
            hourly_component,
            hourly_peaks,
            hourly_peak_contributions,
        ) = self._generate_hourly_component()

        local_noise = self._generate_local_noise()

        demand_propensity = self._build_demand_propensity(
            daily_component=daily_component,
            hourly_component=hourly_component,
            local_noise=local_noise,
        )

        discount_weights = self._build_discount_weights(
            demand_propensity=demand_propensity,
        )

        discount_matrix = self._generate_discount_matrix(
            coverage_matrix=C,
            discount_weights=discount_weights,
            discount_total=discount_total,
        )

        initial_demand = C - discount_matrix

        self._validate_output(
            coverage_matrix=C,
            discount_matrix=discount_matrix,
            initial_demand=initial_demand,
            discount_total=discount_total,
        )

        return DemandNoiseResult(
            initial_demand=initial_demand,
            discount_matrix=discount_matrix,
            demand_propensity=demand_propensity,
            daily_component=daily_component,
            hourly_component=hourly_component,
            local_noise=local_noise,
            discount_weights=discount_weights,
            daily_peak_contributions=daily_peak_contributions,
            hourly_peak_contributions=hourly_peak_contributions,
            k_effective=k_effective,
            discount_total=discount_total,
            daily_peaks=daily_peaks,
            hourly_peaks=hourly_peaks,
        )

    # ============================================================
    # Validaciones
    # ============================================================

    def _validate_init_params(self) -> None:
        if not isinstance(self.k, (int, float)) or self.k < 0:
            raise ValueError("k debe ser un número mayor o igual a 0.")

        if (
            not isinstance(self.max_daily_peaks, int)
            or not 0 <= self.max_daily_peaks <= 4
        ):
            raise ValueError("max_daily_peaks debe ser un entero entre 0 y 4.")

        if (
            not isinstance(self.max_hourly_peaks, int)
            or not 0 <= self.max_hourly_peaks <= 2
        ):
            raise ValueError("max_hourly_peaks debe ser un entero entre 0 y 2.")

        for name, value in {
            "sigma_lambda": self.sigma_lambda,
            "sigma_alpha": self.sigma_alpha,
            "sigma_u": self.sigma_u,
        }.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} debe ser un número mayor o igual a 0.")

        if not isinstance(self.epsilon, (int, float)) or self.epsilon <= 0:
            raise ValueError("epsilon debe ser un número positivo.")

        if not isinstance(self.chi_square_c, (int, float)) or self.chi_square_c <= 0:
            raise ValueError("chi_square_c debe ser un número positivo.")

        if not isinstance(self.q_baseline, (int, float)) or self.q_baseline < 0:
            raise ValueError("q_baseline debe ser un número mayor o igual a 0.")

        if (
            not isinstance(self.capacity_gamma, (int, float))
            or self.capacity_gamma <= 0
        ):
            raise ValueError("capacity_gamma debe ser un número positivo.")

        if (
            not isinstance(self.min_capacity_factor, (int, float))
            or self.min_capacity_factor < 0
            or self.min_capacity_factor > 1
        ):
            raise ValueError("min_capacity_factor debe estar entre 0 y 1.")

    @staticmethod
    def _validate_coverage_matrix(coverage_matrix: np.ndarray) -> np.ndarray:
        C = np.asarray(coverage_matrix)

        if C.shape != (24, 28):
            raise ValueError("coverage_matrix debe tener shape (24, 28).")

        if not np.issubdtype(C.dtype, np.integer):
            if np.all(np.equal(C, np.floor(C))):
                C = C.astype(int)
            else:
                raise ValueError("coverage_matrix debe contener valores enteros.")

        if np.any(C < 0):
            raise ValueError("coverage_matrix debe contener valores mayores o iguales a 0.")

        return C.astype(int)

    @staticmethod
    def _validate_output(
        coverage_matrix: np.ndarray,
        discount_matrix: np.ndarray,
        initial_demand: np.ndarray,
        discount_total: int,
    ) -> None:
        if np.any(discount_matrix < 0):
            raise RuntimeError("La matriz de descuento tiene valores negativos.")

        if np.any(discount_matrix > coverage_matrix):
            raise RuntimeError("La matriz de descuento supera la cobertura en alguna celda.")

        if np.any(initial_demand < 0):
            raise RuntimeError("La demanda inicial contiene valores negativos.")

        if int(discount_matrix.sum()) != int(discount_total):
            raise RuntimeError("La suma de discount_matrix no coincide con discount_total.")

    # ============================================================
    # Utilidades de distribución
    # ============================================================

    @staticmethod
    def _build_sigma_choices(
        min_value: float,
        max_value: float,
        step: float,
    ) -> np.ndarray:
        values = np.arange(min_value, max_value + step, step)
        values = values[values <= max_value]
        return values.astype(float)

    @staticmethod
    def _normal_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        """
        Densidad normal completa.

        f(x) = 1 / (sigma * sqrt(2*pi)) * exp(-(x-mu)^2 / (2*sigma^2))
        """

        return (
            1.0 / (sigma * np.sqrt(2.0 * np.pi))
        ) * np.exp(
            -((x - mu) ** 2) / (2.0 * sigma**2)
        )

    @staticmethod
    def _normal_pdf_from_distance(distance: np.ndarray, sigma: float) -> np.ndarray:
        """
        Densidad normal completa usando una distancia ya calculada.
        """

        return (
            1.0 / (sigma * np.sqrt(2.0 * np.pi))
        ) * np.exp(
            -(distance**2) / (2.0 * sigma**2)
        )

    def _lognormal_mean_one(self, sigma: float, size=None) -> np.ndarray:
        """
        Genera una lognormal con media esperada 1.
        """

        mu = -(sigma**2) / 2.0
        return self.rng.lognormal(mean=mu, sigma=sigma, size=size)

    @staticmethod
    def _normalize_mean_one(array: np.ndarray) -> np.ndarray:
        mean_value = array.mean()

        if mean_value <= 0:
            raise RuntimeError("No se puede normalizar un array con media <= 0.")

        return array / mean_value

    # ============================================================
    # k efectivo
    # ============================================================

    def _sample_k_effective(self) -> float:
        """
        Genera k' en [0, k) usando una chi-cuadrado con 2 grados de libertad.

        X ~ ChiSquare(df=2)

        k' = k * X / (X + c)
        """

        if self.k == 0:
            return 0.0

        x = self.rng.chisquare(df=2)

        k_effective = self.k * x / (x + self.chi_square_c)

        return float(k_effective)

    @staticmethod
    def _compute_discount_total(
        coverage_matrix: np.ndarray,
        k_effective: float,
    ) -> int:
        total_coverage = int(coverage_matrix.sum())

        if total_coverage == 0:
            return 0

        return int(round(k_effective * total_coverage))

    # ============================================================
    # Componentes Q_dia, Q_hora y U
    # ============================================================

    def _generate_daily_component(
        self,
    ) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
        """
        Genera Q_dia y guarda la contribución de cada pico diario.

        Cantidad de picos diaria variable:

            L_D ~ Uniforme discreta {0, 1, ..., max_daily_peaks}

        Fórmula:

            Q_dia[d] = q_baseline + sum_r lambda_r * NormalPDF(d; mu_r, sigma_r)

        Si L_D = 0, se devuelve una componente plana.
        """

        days = np.arange(28, dtype=float)

        amount_of_peaks = int(
            self.rng.integers(0, self.max_daily_peaks + 1)
        )

        if amount_of_peaks == 0:
            daily_component = np.ones(28, dtype=float)
            daily_peak_contributions = np.empty((0, 28), dtype=float)
            return daily_component, [], daily_peak_contributions

        q_day = np.full(28, self.q_baseline, dtype=float)

        peaks: list[dict[str, Any]] = []
        raw_contributions: list[np.ndarray] = []

        for peak_id in range(amount_of_peaks):
            mu = float(self.rng.uniform(0.0, 27.0))
            sigma = float(self.rng.choice(self.sigma_choices))
            intensity = float(self._lognormal_mean_one(self.sigma_lambda))

            contribution = intensity * self._normal_pdf(
                x=days,
                mu=mu,
                sigma=sigma,
            )

            q_day += contribution
            raw_contributions.append(contribution)

            peaks.append(
                {
                    "peak_id": peak_id,
                    "mu": mu,
                    "sigma": sigma,
                    "intensity_lambda": intensity,
                    "raw_max_contribution": float(contribution.max()),
                    "raw_sum_contribution": float(contribution.sum()),
                }
            )

        q_day_mean = q_day.mean()

        if q_day_mean <= 0:
            raise RuntimeError("Q_dia tiene media <= 0.")

        daily_component = q_day / q_day_mean

        daily_peak_contributions = np.vstack(
            [
                contribution / q_day_mean
                for contribution in raw_contributions
            ]
        )

        for peak_id, peak in enumerate(peaks):
            peak["normalized_max_contribution"] = float(
                daily_peak_contributions[peak_id].max()
            )
            peak["normalized_sum_contribution"] = float(
                daily_peak_contributions[peak_id].sum()
            )

        return daily_component, peaks, daily_peak_contributions

    def _generate_hourly_component(
        self,
    ) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
        """
        Genera Q_hora y guarda la contribución de cada pico horario.

        Cantidad de picos horaria variable:

            L_H ~ Uniforme discreta {0, 1, ..., max_hourly_peaks}

        Fórmula:

            Q_hora[h] = q_baseline + sum_l alpha_l * NormalPDF(delta_24(h, nu_l); sigma_l)

        Si L_H = 0, se devuelve una componente plana.
        """

        hours = np.arange(24, dtype=float)

        amount_of_peaks = int(
            self.rng.integers(0, self.max_hourly_peaks + 1)
        )

        if amount_of_peaks == 0:
            hourly_component = np.ones(24, dtype=float)
            hourly_peak_contributions = np.empty((0, 24), dtype=float)
            return hourly_component, [], hourly_peak_contributions

        q_hour = np.full(24, self.q_baseline, dtype=float)

        peaks: list[dict[str, Any]] = []
        raw_contributions: list[np.ndarray] = []

        for peak_id in range(amount_of_peaks):
            nu = float(self.rng.uniform(0.0, 24.0))
            sigma = float(self.rng.choice(self.sigma_choices))
            intensity = float(self._lognormal_mean_one(self.sigma_alpha))

            distance = self._cyclic_hour_distance(hours, nu)

            contribution = intensity * self._normal_pdf_from_distance(
                distance=distance,
                sigma=sigma,
            )

            q_hour += contribution
            raw_contributions.append(contribution)

            peaks.append(
                {
                    "peak_id": peak_id,
                    "nu": nu,
                    "sigma": sigma,
                    "intensity_alpha": intensity,
                    "raw_max_contribution": float(contribution.max()),
                    "raw_sum_contribution": float(contribution.sum()),
                }
            )

        q_hour_mean = q_hour.mean()

        if q_hour_mean <= 0:
            raise RuntimeError("Q_hora tiene media <= 0.")

        hourly_component = q_hour / q_hour_mean

        hourly_peak_contributions = np.vstack(
            [
                contribution / q_hour_mean
                for contribution in raw_contributions
            ]
        )

        for peak_id, peak in enumerate(peaks):
            peak["normalized_max_contribution"] = float(
                hourly_peak_contributions[peak_id].max()
            )
            peak["normalized_sum_contribution"] = float(
                hourly_peak_contributions[peak_id].sum()
            )

        return hourly_component, peaks, hourly_peak_contributions

    def _generate_local_noise(self) -> np.ndarray:
        """
        Genera U[h,d] lognormal con media esperada 1.
        """

        U = self._lognormal_mean_one(
            sigma=self.sigma_u,
            size=(24, 28),
        )

        U = self._normalize_mean_one(U)

        return U

    @staticmethod
    def _cyclic_hour_distance(hours: np.ndarray, center: float) -> np.ndarray:
        """
        Distancia cíclica horaria.

        delta_24(h, nu) = min(|h - nu|, 24 - |h - nu|)
        """

        raw_distance = np.abs(hours - center)
        return np.minimum(raw_distance, 24.0 - raw_distance)

    # ============================================================
    # Q y pesos de descuento
    # ============================================================

    def _build_demand_propensity(
        self,
        daily_component: np.ndarray,
        hourly_component: np.ndarray,
        local_noise: np.ndarray,
    ) -> np.ndarray:
        """
        Construye Q[h,d]:

            Q[h,d] = Q_hora[h] * Q_dia[d] * U[h,d]
        """

        Q = (
            hourly_component[:, None]
            * daily_component[None, :]
            * local_noise
        )

        Q = self._normalize_mean_one(Q)

        return Q

    def _build_discount_weights(
        self,
        demand_propensity: np.ndarray,
    ) -> np.ndarray:
        """
        Convierte Q en pesos de descuento mediante inversión min-max.

        Q representa propensión deseada de demanda:
            Q alto -> preservar más demanda -> menor descuento
            Q bajo -> preservar menos demanda -> mayor descuento

        Fórmulas:
            Q_norm = (Q - min(Q)) / (max(Q) - min(Q) + epsilon)

            W = (1 - Q_norm) + epsilon

        Luego W se normaliza a media 1.
        """

        q_min = float(demand_propensity.min())
        q_max = float(demand_propensity.max())

        q_norm = (demand_propensity - q_min) / (
            q_max - q_min + self.epsilon
        )

        weights = (1.0 - q_norm) + self.epsilon

        weights = self._normalize_mean_one(weights)

        return weights

    # ============================================================
    # Generación de matriz R
    # ============================================================

    def _generate_discount_matrix(
        self,
        coverage_matrix: np.ndarray,
        discount_weights: np.ndarray,
        discount_total: int,
    ) -> np.ndarray:
        """
        Genera la matriz de descuento R de forma secuencial.

        En cada iteración r se calcula un score de selección:

            score[h,d] = W[h,d] * F_cap[h,d]

        donde:
            W[h,d] = peso de descuento derivado de la propensión de demanda.
            F_cap[h,d] = factor suavizado de capacidad restante.

        La capacidad restante no entra de forma lineal. Primero se normaliza:

            cap_ratio[h,d] = remaining_capacity[h,d] / max(remaining_capacity)

        y luego se suaviza:

            F_cap[h,d] = min_capacity_factor
                         + (1 - min_capacity_factor)
                         * cap_ratio[h,d] ** capacity_gamma

        Para celdas sin capacidad restante, F_cap[h,d] = 0.

        Esto reduce el dominio de las celdas con mucha capacidad disponible y
        permite que los pesos de propensión tengan mayor influencia.
        """

        C = coverage_matrix.astype(int)

        if discount_total == 0:
            return np.zeros_like(C, dtype=int)

        capacity = C.copy()

        if discount_total > int(capacity.sum()):
            raise ValueError("discount_total supera la capacidad total disponible.")

        R = np.zeros_like(C, dtype=int)

        for _ in range(discount_total):
            remaining_capacity = capacity - R

            max_remaining_capacity = int(remaining_capacity.max())

            if max_remaining_capacity <= 0:
                raise RuntimeError("No quedan celdas disponibles para descontar.")

            cap_ratio = remaining_capacity / max_remaining_capacity

            capacity_factor = self.min_capacity_factor + (
                1.0 - self.min_capacity_factor
            ) * (cap_ratio ** self.capacity_gamma)

            capacity_factor = np.where(
                remaining_capacity > 0,
                capacity_factor,
                0.0,
            )

            selection_score = discount_weights * capacity_factor

            total_score = selection_score.sum()

            if total_score <= 0:
                raise RuntimeError("No quedan celdas disponibles para descontar.")

            probabilities = (selection_score / total_score).ravel()

            selected_flat_idx = int(
                self.rng.choice(
                    probabilities.size,
                    p=probabilities,
                )
            )

            h, d = np.unravel_index(
                selected_flat_idx,
                C.shape,
            )

            R[h, d] += 1

        return R
# ============================================================
# Ejemplo mínimo de uso
# ============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(1)

    coverage_matrix = rng.integers(
        low=0,
        high=6,
        size=(24, 28),
    )

    noise_generator = DemandNoiseGenerator(
        k=0.20,
        max_daily_peaks=4,
        max_hourly_peaks=2,
        sigma_lambda=0.50,
        sigma_alpha=0.50,
        sigma_u=0.15,
        epsilon=1e-9,
        chi_square_c=4.0,
        seed=42,
    )

    result = noise_generator.generate(coverage_matrix)

    print("Cobertura total:", int(coverage_matrix.sum()))
    print("Descuento total:", int(result.discount_matrix.sum()))
    print("Demanda inicial total:", int(result.initial_demand.sum()))
    print("k efectivo:", result.k_effective)
    print("daily_component shape:", result.daily_component.shape)
    print("hourly_component shape:", result.hourly_component.shape)
    print("daily_peak_contributions shape:", result.daily_peak_contributions.shape)
    print("hourly_peak_contributions shape:", result.hourly_peak_contributions.shape)
    print("D0 >= 0:", bool(np.all(result.initial_demand >= 0)))
    print("R <= C:", bool(np.all(result.discount_matrix <= coverage_matrix)))

    print("\nPicos diarios:")
    if result.daily_peaks:
        for peak in result.daily_peaks:
            print(
                f"peak_id={peak['peak_id']} | "
                f"mu={peak['mu']:.2f} | "
                f"sigma={peak['sigma']:.2f} | "
                f"lambda={peak['intensity_lambda']:.4f}"
            )
    else:
        print("Sin picos diarios.")

    print("\nPicos horarios:")
    if result.hourly_peaks:
        for peak in result.hourly_peaks:
            print(
                f"peak_id={peak['peak_id']} | "
                f"nu={peak['nu']:.2f} | "
                f"sigma={peak['sigma']:.2f} | "
                f"alpha={peak['intensity_alpha']:.4f}"
            )
    else:
        print("Sin picos horarios.")