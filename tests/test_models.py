import pytest

from src.common.models import BoardObservation
from src.common.squares import ALL_SQUARES

START_OCCUPIED = {s for s in ALL_SQUARES if s[1] in "1278"}


def make_obs(occupied_set=START_OCCUPIED, **kwargs):
    params = dict(
        occupied={s: s in occupied_set for s in ALL_SQUARES},
        confidence={s: 0.9 for s in ALL_SQUARES},
        stable=True,
    )
    params.update(kwargs)
    return BoardObservation(**params)


def test_valid_observation():
    obs = make_obs(changed_squares={"e2", "e4"}, frame_id=3, timestamp=12.5)
    assert len(obs.occupied) == 64
    assert obs.occupied["e2"] is True
    assert obs.occupied["e4"] is False
    assert obs.changed_squares == frozenset({"e2", "e4"})
    assert len(obs.occupied_squares) == 32


def test_e2e4_example():
    before = make_obs()
    after = make_obs(START_OCCUPIED - {"e2"} | {"e4"}, changed_squares={"e2", "e4"}, frame_id=1)
    assert before.occupied["e2"] and not before.occupied["e4"]
    assert not after.occupied["e2"] and after.occupied["e4"]
    assert before.occupied_squares ^ after.occupied_squares == {"e2", "e4"}


def test_observation_is_immutable():
    obs = make_obs()
    with pytest.raises(TypeError):
        obs.occupied["e4"] = True
    with pytest.raises(AttributeError):
        obs.stable = False


def test_observation_copies_input():
    occupied = {s: False for s in ALL_SQUARES}
    obs = make_obs(occupied=occupied)
    occupied["e4"] = True
    assert obs.occupied["e4"] is False


def test_missing_square_rejected():
    occupied = {s: False for s in ALL_SQUARES}
    del occupied["h8"]
    with pytest.raises(ValueError, match="h8"):
        make_obs(occupied=occupied)


def test_extra_square_rejected():
    confidence = {s: 0.5 for s in ALL_SQUARES}
    confidence["i9"] = 0.5
    with pytest.raises(ValueError, match="i9"):
        make_obs(confidence=confidence)


@pytest.mark.parametrize("value", [1.5, -0.1, float("nan")])
def test_confidence_out_of_range(value):
    confidence = {s: 0.5 for s in ALL_SQUARES}
    confidence["e4"] = value
    with pytest.raises(ValueError):
        make_obs(confidence=confidence)


def test_occupied_must_be_bool():
    occupied = {s: False for s in ALL_SQUARES}
    occupied["e4"] = 1
    with pytest.raises(TypeError):
        make_obs(occupied=occupied)


def test_bad_changed_square_rejected():
    with pytest.raises(ValueError):
        make_obs(changed_squares={"e4", "z0"})


def test_changed_squares_string_rejected():
    with pytest.raises(TypeError):
        make_obs(changed_squares="e4")


def test_negative_frame_id_rejected():
    with pytest.raises(ValueError):
        make_obs(frame_id=-1)


def test_unstable_observation_allowed():
    obs = make_obs(stable=False)
    assert obs.stable is False
