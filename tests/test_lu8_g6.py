from genny.physical import winds as W


def test_trumpet_optimal_blowing_pressure_adachi_sato_p10():
    kpa = {232.0: 2.5, 341.0: 3.0, 457.0: 3.5, 572.0: 4.0, 686.0: 5.0}
    for fpeak, _flip, _fsound, p0 in W.TRUMPET_OPTIMAL:
        assert abs(p0 - 1000.0 * kpa[fpeak]) < 1e-6
