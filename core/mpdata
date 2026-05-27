import numpy as np
from PyMPDATA_examples.Magnuszewski_et_al_2025.asian_option import AsianArithmetic, Settings
from PyMPDATA_examples.Magnuszewski_et_al_2025.common import OPTIONS

def assert_the_stability_condition_for_mpdata(S_min, S_max, nx, sgma, T, nt):
    log_s_min = np.log(S_min)
    log_s_max = np.log(S_max)
    dx = (log_s_max - log_s_min) / nx
    sigma_squared = sgma ** 2
    dt = T / nt

    l2 = dx * dx / sigma_squared / dt
    if (l2 <= 2):
        print(f"Error: MPDATA: Lambda squared should be more than 2 for stability {l2}")
    return l2 > 2

def calculate_mpdata_price(T, K, r, sgma, S_max, S_min, nt, nx, ny, scale, S0, option="MPDATA (2 it.)"):
    assert_the_stability_condition_for_mpdata(S_min=S_min, S_max=S_max, nx=nx, sgma=sgma, T=T, nt=nt)

    SETTINGS = Settings(
        T=T,
        K=K,
        r=r,
        sgma=sgma,
        S_max=S_max,
        S_min=S_min,
    )

    # Set the resolution (Scale 4 or 8 provides high accuracy)
    res = {
        'nt': nt * scale + 1,
        'nx': nx * scale + 1,
        'ny': ny * scale + 1,
    }

    sim = AsianArithmetic(SETTINGS, **res, options=OPTIONS[option], variant="call")

    print(f"Running simulation for {res['nt']} steps...")
    sim.step(sim.nt)

    # The average dimension (A) starts at 0 at index 0
    prices_at_t0 = sim.solver.advectee.get()[:, 0]

    final_price = np.interp(S0, sim.S, prices_at_t0)

    return final_price

def mpdata_solution(T, K, r, sgma, S_max, S_min, nt, nx, ny, scale, S0):

    final_price = calculate_mpdata_price(T, K, r, sgma, S_max, S_min, nt, nx, ny, scale, S0)

    print("-" * 30)
    print(f"MPDATA Asian Option Price (S0={S0}): {final_price:.4f}")
    print("-" * 30)
