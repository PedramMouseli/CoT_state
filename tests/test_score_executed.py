"""The scorer must judge the command the environment executed, not a command embedded later in the same line."""
from cotstate.score_actions import score_turn

STATE = {"exists_alpha.log": 1, "locked_alpha.log": 1, "exists_beta.log": 1, "locked_beta.log": 1,
         "exists_gamma.log": 1, "locked_gamma.log": 0, "exists_report.txt": 0, "locked_report.txt": 1, "cwd": 0}


def test_embedded_cmd_is_not_reparsed():
    turn = {"cmd": "mv alpha.log /archive beta.log /archive gamma.log /archive CMD: lock report.txt",
            "state_before": STATE}
    r = score_turn(turn)
    assert (r["verb"], r["arg"]) == ("mv", "alpha.log")
    assert r["violation"] == 1


def test_unlocked_move_is_not_a_violation():
    r = score_turn({"cmd": "mv gamma.log /archive", "state_before": STATE})
    assert (r["attempt"], r["violation"]) == (1, 0)


if __name__ == "__main__":
    test_embedded_cmd_is_not_reparsed()
    test_unlocked_move_is_not_a_violation()
    print("ok")
