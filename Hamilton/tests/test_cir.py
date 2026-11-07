"""
Phase 5f tests: channel assembly (spec §12.7, eq. 12.5), the received-signal helper for FMCW sweeps and pulse
trains, and the output writer (§14.2, npz + JSON).
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import numpy as np

from receiver import cir
from core.constants import C
from propagation.legs import L_POL_DB
from core.records import LegMode
from echoes.twoway import target_component

F0 = 8e6


def _leg(label, Pg, P, doppler, site="tx"):
    return LegMode(site=site, mode="O", label=label, n_hops=1, beta=0.4, alpha=0.0,
                   e_site=np.array([1.0, 0, 0]), k_far=np.array([1.0, 0, 0]), grazing_far=0.3,
                   Pg=Pg, P=P, delay=Pg / C, doppler=doppler, absorption_dB=8.0, absorption_D_dB=3.0,
                   ground_loss_dB=0.0, d_eff=1000e3, caustic_flag=False, xmode_validated=None,
                   attenuation_flag=False)


def _component(Pg_out=0.3e6, Pg_ret=0.33e6, doppler_out=0.3, doppler_ret=0.2, v=(0.0, 0.0, 0.0)):
    out = _leg("1F-low-O", Pg_out, Pg_out - 5e3, doppler_out)
    ret = _leg("1F-low-O", Pg_ret, Pg_ret - 5e3, doppler_ret, site="rx")
    return target_component(out, ret, f0=F0, v_T=np.array(v, dtype=float), k_i=np.array([1.0, 0, 0]),
                            s_f=np.array([1.0, 0, 0]), sigma_T_m2=10.0, Pt_dBW=40.0)


def test_single_target_impulse_response_sits_at_its_delay_with_its_phase_and_doppler():
    comp = _component(v=(0.0, 0.0, 0.0))
    dtau = 2e-5
    tau = np.arange(0.0, 4e-3, dtau)
    t = np.arange(0.0, 0.2, 1e-3)
    h = cir.assemble_cir([comp], t, tau)
    k_peak = np.argmax(np.abs(h[0]))
    assert abs(tau[k_peak] - comp.delay) <= dtau / 2 + 1e-12
    expect_amp = np.sqrt(comp.power) * np.exp(1j * comp.phase)
    k_c = int(round((comp.delay - tau[0]) / dtau))
    # band-limited delta: the value on the bin is the amplitude times sinc of the fractional offset
    expect_bin = expect_amp * np.sinc((tau[k_c] - comp.delay) / dtau)
    assert abs(h[0, k_c] - expect_bin) < 1e-12 * abs(expect_amp)
    ratio = h[10, k_c] / h[0, k_c]
    assert abs(ratio - np.exp(2j * np.pi * comp.doppler * t[10])) < 0.05


def test_received_fmcw_sweep_is_a_delayed_and_doppler_shifted_copy():
    comp = _component(Pg_out=0.3e6, Pg_ret=0.3e6, doppler_out=0.0, doppler_ret=0.0)   # delay 2 ms
    comp.doppler = 1.0                                                              # Hz, set explicitly
    dtau = 2e-5
    tau = np.arange(0.0, 4.5e-3, dtau)
    t = np.arange(0.0, 0.2, 1e-4)
    h = cir.assemble_cir([comp], t, tau)
    T, B = 0.02, 10e3
    x = lambda s: cir.fmcw_sweep(s, B, T)                                           # noqa: E731
    y = cir.received_signal(h, t, tau, x)
    amp = np.sqrt(comp.power) * np.exp(1j * comp.phase)
    expect = amp * np.exp(2j * np.pi * comp.doppler * t) * x(t - comp.delay)
    inside = slice(int(0.03 / 1e-4), int(0.15 / 1e-4))
    err = np.max(np.abs(y[inside] - expect[inside])) / abs(amp)
    assert err < 0.05, err


def test_pulse_train_helper_and_noise_addition():
    s = np.arange(0.0, 1.0, 1e-4)
    p = cir.pulse_train(s, 1e-3, 0.1, 5)
    assert abs(np.sum(p.real) * 1e-4 - 5e-3) < 2e-4                                   # five 1 ms pulses
    rng = np.random.default_rng(0)
    tau = np.arange(0.0, 1e-3, 1e-5)
    t = np.arange(0.0, 0.05, 1e-3)
    h = np.zeros((t.size, tau.size), complex)
    y = cir.received_signal(h, t, tau, lambda s: np.ones_like(s), noise_power_W=2.0, rng=rng)
    assert abs(np.mean(np.abs(y) ** 2) / 2.0 - 1.0) < 0.2


def test_writer_round_trip_and_json_has_no_nan():
    comp = _component()
    tau = np.arange(0.0, 2e-3, 2e-5)
    t = np.arange(0.0, 0.01, 1e-3)
    h = cir.assemble_cir([comp], t, tau)
    with tempfile.TemporaryDirectory() as d:
        npz, js = cir.write_cir(pathlib.Path(d) / "run", h=h, t=t, tau_grid=tau, comps=[comp], f0=F0,
                                models={"ionosphere": "E-CHAIM+IRI-2020 (test)"}, seeds={"noise": 0})
        z = np.load(npz)
        assert np.allclose(z["h"], h) and np.allclose(z["tau"], tau)
        doc = json.loads(js.read_text())
        assert doc["carrier_Hz"] == F0 and len(doc["components"]) == 1 and len(doc["legs"]) == 2
        assert "NaN" not in js.read_text()
        assert doc["components"][0]["gamma_or"] is None and doc["provenance"]["code_version"]
