"""Configuration dataclasses for the G1 locomotion environment."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainRandCfg:
    """Ranges for the reset()-time domain-randomization pipeline."""

    payload_mass_range: tuple[float, float] = (0.0, 5.0)  # kg, added to torso
    payload_com_offset_range: tuple[float, float] = (-0.05, 0.05)  # m, per-axis torso CoM shift
    friction_range: tuple[float, float] = (0.4, 1.2)
    terrain_enabled: bool = False  # flat floor while debugging/early training; flip on once it can stand/walk
    terrain_res: int = 8  # Perlin gradient-cell count; higher = bumpier terrain (must divide 128)
    push_interval_range: tuple[float, float] = (3.0, 5.0)  # seconds between pushes
    push_lin_vel_range: tuple[float, float] = (-0.5, 0.5)  # m/s impulse kick per axis (x, y)
    init_joint_noise: float = 0.1  # rad, uniform noise added to default pose on reset
    init_yaw_range: tuple[float, float] = (-3.14159, 3.14159)


@dataclass
class RewardScales:
    """Signed weights for every reward term. Positive = reward, negative = penalty.

    Terms are grouped below by where the value came from: matched to Unitree's official G1 configs,
    project-specific additions, or values tuned by this project's own run history.

    v10 change set (this revision) vs v9_smoke, in one place:
      - contact_timing (0.5, air-time based) DELETED -> replaced by clocked `contact` (+0.18)
      - feet_clearance (+1.0 reward)         INVERTED -> feet_swing_height (-20.0 penalty)
      - ADDED: hip_pos, dof_pos_limits, dof_acc, dof_vel, contact_no_vel, waist_dev, arm_dev
      - unchanged: everything else, including alive_bonus (deliberately held at 0.5 this run so it
        is not a confound while the cadence/clearance terms are being evaluated)
    """

    # =========================================================================================
    # v12: REVERTED TO v9's REWARD SCALES. The gait clock is the ONLY intended change.
    #
    # Why. Ranked by locomotion quality rather than survival (16 envs x 2000 steps each):
    #
    #   run   ep_len   tilt    vel err   both feet down   contact flips/s
    #   v3     110.1   12.7      0.752       82.6%             6.74
    #   v7      55.5   18.6      0.677       23.7%             9.33
    #   v9      65.4   15.0     [0.657]      34.4%            17.25
    #   rand    40.1   11.9      0.903       49.4%            18.52
    #
    # v3 has by far the longest episodes and is the WORST policy: 82.6% double support and
    # velocity tracking barely better than an untrained network. alive_bonus=5.0 bought a statue.
    # v9 tracks best and is the only one that visibly strides. Episode length rewards standing
    # still, so it must not be used alone to rank these.
    #
    # Every reward change after v9 made things worse -- v10 (43 peak, suicide policy) and v11
    # (18, no signal). Both were justified by matching published reference configs. Those configs
    # are tuned as a PACKAGE against Isaac Gym's model, actuators and ~1B-timestep budget; lifting
    # individual weights out of one and into a different sim, a 29-DoF arm-actuated robot and a
    # ~12M-timestep budget does not carry the tuning with them. So: back to the values with
    # evidence behind them, and change exactly one thing.
    # =========================================================================================

    # --- velocity tracking (the actual task) -------------------------------------------------
    # Unitree uses 1.0 / 0.5; we run 1.5 / 0.75 -- same 2:1 ratio, scaled up. Unchanged since v9.
    lin_vel_tracking: float = 1.5
    ang_vel_tracking: float = 0.75

    # --- gait structure ----------------------------------------------------------------------
    # Clocked contact reward, matching unitree_rl_gym's G1 `contact` term: a fixed alternating
    # gait clock says which foot *should* be in stance right now, and this pays out per foot whose
    # actual contact state matches. Range 0..2 feet -> 0..+0.36 per step.
    #
    # This REPLACES the old air-time-based `contact_timing` (0.5). That term only paid out after a
    # complete, well-timed swing-and-land cycle, with no schedule telling the policy when a swing
    # was due -- so "never lift a foot" stayed locally optimal (the v8_smoke freeze) and, later,
    # nothing pinned down cadence (the v9/model_4750 over-long strides). Note the weight drop from
    # 0.5 to 0.18: it fires far more often (every step, not once per completed cycle), so the
    # per-step contribution is comparable despite the smaller scale.
    #
    # *** THE ONE INTENDED CHANGE IN v12 ***
    #
    # Clocked contact reward, replacing v9's air-time `contact_timing` (which was also 0.5, so the
    # weight is unchanged -- only the criterion differs). v9's version paid out for a foot that had
    # been airborne a while, with no schedule saying WHEN a swing was due. The result is visible in
    # the measurement above: v9 runs 17.25 contact changes/s against the ~2.5/s a real 0.8 s gait
    # implies -- 7x too fast. That is foot chatter, not stepping, and it is why v9 strides but
    # scuffs. A fixed alternating clock is the standard fix and is what both Unitree repos use
    # (unitree_rl_lab's `gait` term is weighted 0.5 at period 0.8 / offset 0.5 / duty 0.55 -- the
    # same clock this implements).
    #
    # This has never actually been tested. v10 carried it, but v10's episodes averaged 0.42 s
    # against a 0.8 s gait period, so the policy never observed a full cycle and `reward/contact`
    # never escaped its 0.18/step chance level. v9 survives ~1.31 s, comfortably longer than one
    # period, so the clock can engage for the first time here.
    contact: float = 0.5
    # Swing-foot clearance, POSITIVE reward -- v9's exact form and weight (target 0.08, see
    # FOOT_CLEARANCE_TARGET in rewards.py). v10 inverted this into a -20.0 penalty taken from the
    # 12-DoF unitree_rl_gym; unitree_rl_lab's 29dof config keeps it a +1.0 reward, and v9's own
    # result -- the only policy that visibly strides -- was produced with the reward form.
    feet_clearance: float = 1.0
    # Both feet simultaneously airborne. Project-specific (Unitree has no equivalent -- the clock
    # covers it for them). At stance duty 0.55 with a 0.5 phase offset the clock schedules ~10%
    # double support and 0% double flight, so under good clock tracking this term should read ~0;
    # it exists to punish deviation *from* the clock, i.e. genuine hopping.
    double_flight: float = -0.3
    # Soft support-polygon margin. Project-specific. Verified NOT to be a statically-stable
    # criterion: it only charges once the CoP is more than SUPPORT_RADIUS (0.12 m) from the foot
    # center (single support) or from the line between the feet (double support). A G1 foot is
    # ~0.22 m long, so a CoP at the toe during push-off (~0.11 m from center) is inside the free
    # radius and costs nothing. It penalizes "CoP outside the foot entirely", not "CoP near the
    # edge", so it does not fight dynamic walking.
    zmp_margin: float = -0.2
    torque_penalty: float = -1.0e-5
    action_rate_penalty: float = -0.01  # v9 value; v11's -0.05 reverted with the rest

    # --- v10/v11 ADDITIONS, ALL DISABLED FOR v12 ---------------------------------------------
    # Kept at 0.0 rather than deleted: each is implemented and logged, so re-enabling one is a
    # one-line change, and per-term logging still shows what it WOULD have charged (terms are
    # computed then multiplied by their weight, so a 0.0 weight logs 0.0 -- flip the weight to see
    # it). These went in as a block in v10 alongside the reward-scale changes that caused the
    # suicide policy, so none of them has ever been evaluated on its own. Re-enable one at a time,
    # each against the v12 baseline, and keep only what beats it.
    #
    # My prior on which are worth trying first, for whenever that happens: hip_pos (Unitree state
    # its purpose is preventing the splayed-leg gait, which is a real failure mode here) and
    # contact_no_vel (anti-scuff, and v9's chatter is exactly a scuffing problem).
    energy: float = 0.0

    # --- joint regularization (all NEW in this change set) -----------------------------------
    # Hip roll/yaw pinned near default. Unitree's `hip_pos`, their stated purpose being to prevent
    # the splayed-leg gait RL converges on. Hip pitch is deliberately NOT included.
    hip_pos: float = 0.0
    # Soft joint-limit violation, Unitree's `dof_pos_limits` at soft_dof_pos_limit=0.9 (see
    # EnvCfg.soft_dof_pos_limit). Requires the crouched default pose to have landed first -- at the
    # old all-zero default the knee sat outside its own soft lower bound (see robot.py).
    dof_pos_limits: float = 0.0
    # Joint acceleration / velocity smoothness, Unitree's values. NOTE: Unitree applies these over
    # 12 leg joints, we apply over all 29, so the same weight accumulates ~2.4x more sum-of-squares
    # here. Flagged as a retune candidate if the negatives turn out to dominate.
    dof_acc: float = 0.0
    dof_vel: float = 0.0
    # Foot moving while in contact -- the anti-slip / anti-scuff term. Unitree's `contact_no_vel`.
    # Mostly a sim2real term, but it also directly discourages the foot-drag behavior.
    contact_no_vel: float = 0.0

    # --- upper-body regularization (NEW, project-specific) -----------------------------------
    # No Unitree counterpart exists: unitree_rl_gym's G1 is the 12-DoF build with the arms FIXED in
    # the URDF, so it needs no arm terms at all. Ours has 3 waist + 14 arm/wrist actuated joints
    # with zero counter-pressure, and the v9 policy is visibly using them to generate angular
    # momentum instead of walking.
    #
    # Weights chosen as: waist == hip_pos (-1.0), arms deliberately softer (-0.25).
    #   - Waist gets the hip_pos weight because it plays the same role for the torso that hip
    #     roll/yaw plays for the legs (keep it square), and unitree_rl_lab's 29-DoF config
    #     describes penalizing waist heavily. Only 3 joints, both roll/pitch limited to +-0.52 rad,
    #     so the worst case is bounded around -0.8/step.
    #   - Arms are 14 joints, so an equal per-joint weight would let their sum-of-squares dominate
    #     the waist term ~4.7x. Arm swing is also legitimate biped balance behavior -- the goal is
    #     removing the *free* angular-momentum exploit, not freezing the arms into a mannequin.
    #     At -0.25 a typical 0.3 rad average deviation costs ~-0.32/step.
    # v11: both now have a confirmed reference. unitree_rl_lab's 29dof config DOES carry
    # upper-body deviation terms (the "no counterpart exists" note above was written from
    # unitree_rl_gym's 12-DoF config only):
    #     joint_deviation_waists = RewTerm(joint_deviation_l1, weight=-1,   joint_names=["waist.*"])
    #     joint_deviation_arms   = RewTerm(joint_deviation_l1, weight=-0.1,
    #                                      joint_names=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*"])
    # Waist -1.0 was already correct by coincidence. Arms move -0.25 -> -0.1 to match.
    # Note their form is joint_deviation_l1 (absolute), ours is squared; see rewards.py.
    waist_deviation: float = 0.0
    arm_deviation: float = 0.0
    # History: asymmetric_payload_run_v2 (5000 iters) plateaued at ~64/1000 step episodes (6.4%
    # of max) with alive_bonus=0.1 -- too small relative to the penalty terms to give PPO much
    # incentive to fight for extra survival time. Bumped to 5.0 for v3/v4/v5 based on a comparable
    # open-source G1 policy (https://huggingface.co/hardware-pathon-ai/unitree-g1-phase1-locomotion)
    # that used that value -- but that repo is Isaac-Gym Phase-1 standing-only (frozen arms, no
    # velocity-tracking goal), not a walking policy. Checking Unitree's own *official* G1
    # velocity-locomotion config (the exact robot, the exact task) instead:
    # https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py
    # -- their alive weight is 0.15, ~33x smaller than our 5.0. At 5.0, alive_bonus dominates the
    # per-step reward over lin_vel_tracking/ang_vel_tracking (confirmed in asymmetric_payload_run_v3's
    # logs -- see PROGRESS.md), which plausibly explains why entropy/action-std climbed unchecked
    # even with entropy_coef=0.01 matching Unitree's own value exactly: with alive_bonus this
    # dominant, there's little reward cost to noisy/imprecise actions to counteract entropy_coef's
    # upward pressure. Set to Unitree's own value for asymmetric_payload_run_v6.
    # v6/v7 (both at 0.15) plateaued at only ~5.5-6.4% max episode length -- worse than v2's own
    # 0.1 result -- once contact_timing/feet_clearance were also added as competing per-step terms
    # (see PROGRESS.md's asymmetric_payload_run_v7 analysis): with alive_bonus this small, there's
    # too little reward gradient toward simply not falling before those other terms can pay out.
    # Bumped to 1.0 for asymmetric_payload_run_v8_smoke -- comparable in magnitude to
    # feet_clearance (1.0) and lin_vel_tracking (1.5) rather than either dominating (5.0 did) or
    # being swamped (0.15 was) by them. Killed early: plateaued hard at ~66/1000 step episodes
    # (contact_timing pinned at ~0, confirmed visually in play.py -- the policy locked into
    # standing perfectly rigid, "freezing" its way to alive_bonus rather than ever stepping, then
    # slowly toppling once instability accumulates) for 500+ iterations with zero further
    # improvement -- alive_bonus at 42% of per-step reward (vs 6.9% in v7) was still enough to
    # make "don't move" the reward-maximizing strategy. Trying 0.5 next.
    #
    # v9_smoke result (0.5): the freeze failure mode is GONE. model_4750/model_4999 take long
    # alternating strides -- they do not stand rigid any more. The remaining failure is different:
    # over-long strides, a dragging trailing foot, and accumulating forward pitch until it topples.
    # That is a cadence + clearance problem, not an alive_bonus problem, which is why this change
    # set targets the gait clock and swing-height terms instead. alive_bonus is deliberately HELD
    # at 0.5 for this run so it is not a confound while those are evaluated -- do not retune it and
    # the new terms in the same run.
    alive_bonus: float = 0.5
    # --- stability (unchanged, all matching Unitree exactly) ----------------------------------
    # Stability terms below match Unitree's official G1 RL config (see rewards.py for the
    # per-term explanation of what each one physically means for the robot):
    # https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py
    # v11: -1.0 -> -5.0. -1.0 was unitree_rl_gym (12-DoF); unitree_rl_lab's 29dof config uses
    # `flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)`. Same quantity
    # (squared xy of projected gravity), 5x the weight. This is also the term most directly aimed
    # at v10's failure: terminations were 100% orientation, i.e. forward pitch.
    orientation: float = -1.0  # v9 value; v11 raised it to -5.0 (unitree_rl_lab). Reverted.
    base_height: float = -10.0
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    # NOTE on dof_acc / dof_vel / ang_vel_xy: I earlier proposed cutting these to 0.25x, derived
    # from a local -35.3/step random-action measurement. That was NOT a referenced value and has
    # been dropped. unitree_rl_lab's 29dof config applies joint_vel -0.001 and joint_acc -2.5e-7
    # UNSCOPED, i.e. over all 29 joints of this exact robot -- so the "we have 29 joints, they have
    # 12" argument for scaling them down was simply wrong. The real discrepancy was the missing
    # control_dt factor; see REWARD_DT in rewards.py.


@dataclass
class EnvCfg:
    scene_xml: str = "assets/g1/scene_train.xml"
    num_envs: int = 2
    device: str = "cpu"
    control_dt: float = 0.02  # 50 Hz policy rate
    sim_dt: float = 0.005  # 200 Hz physics rate (4 substeps per control step)
    max_episode_length_s: float = 20.0

    # --- termination ---------------------------------------------------------------------------
    # Was 0.5 m. At a 0.784 m standing height that terminated on ordinary gait-cycle pelvis dips,
    # not just genuine falls -- and it gets worse with the crouched default, which already sits
    # lower. Relaxed to Unitree's 0.2 m, i.e. "the base is essentially on the floor".
    fall_height_threshold: float = 0.2
    # Terminate once the torso tilts past this angle from vertical. Stored as the projected-gravity
    # z-component threshold actually used at the comparison site: |g_z| = cos(tilt), so the config
    # value is cos(0.8 rad) = 0.6967. Was 0.6 (= 53.1 deg); tightened to Unitree's 0.8 rad
    # (45.8 deg) because our live failure mode is specifically accumulating forward pitch, so the
    # tighter bound cuts those episodes off earlier rather than burning samples on a slow topple.
    fall_gravity_threshold: float = 0.6967
    # Terminate on pelvis-vs-floor contact, matching Unitree's terminate_after_contacts_on=["pelvis"].
    # Needed alongside the relaxed 0.2 m height bound: without it, a robot that has fallen onto its
    # back but is still geometrically above 0.2 m keeps running as a live episode.
    terminate_on_pelvis_contact: bool = True

    # --- gait clock ----------------------------------------------------------------------------
    # Fixed alternating gait schedule driving both the `contact` reward and 2 observation terms.
    # Phase is derived from the episode step counter (phase = t/period mod 1), so it resets with
    # the episode and cannot desync from it.
    #
    # 0.8 s (1.25 Hz) is unitree_rl_lab's G1 period (vs 0.5 s Go2, 0.6 s H1). The right leg runs at
    # phase + 0.5, i.e. exactly antiphase.
    gait_period_s: float = 0.8
    # Fraction of each leg's cycle spent in stance. 0.55, NOT 0.5: at exactly 0.5 duty with a 0.5
    # phase offset the two stance windows never overlap, so there is no double-support phase at all
    # -- that schedules a run, not a walk, and double_flight (-0.3) could never fire under correct
    # tracking. At 0.55 the overlap is 2*0.55-1 = 10% of the cycle in double support, and 0% in
    # double flight. Also matches unitree_rl_gym's G1 contact reward, which tests leg_phase < 0.55.
    gait_stance_duty: float = 0.55

    # Soft joint-limit fraction for the dof_pos_limits penalty, matching Unitree. The penalty-free
    # band is the middle 90% of each joint's MJCF range: [mid - 0.9*half, mid + 0.9*half].
    soft_dof_pos_limit: float = 0.9

    # Clamp the summed per-step reward at >= 0 (legged_gym's `only_positive_rewards`).
    #
    # v11: NOW ON. This is the single most important fix in this change set, and v10 is what it
    # costs to get it wrong.
    #
    # It is the legged_gym base default, verbatim from LeggedRobotCfg.rewards:
    #     only_positive_rewards = True
    #     # if true negative total rewards are clipped at zero
    #     # (avoids early termination problems)
    # Unitree's G1 `rewards` class overrides only soft_dof_pos_limit and base_height_target, so it
    # INHERITS True. Every reference implementation of this task runs with the clamp on.
    #
    # It was set False for v10 on the following measurement (300 steps, disturbances off):
    #   holding the crouched default (stand still):  pos +1.38 / neg -0.35 -> NET +1.02
    #   random actions (where PPO starts):           pos +1.00 / neg -36.3 -> NET -35.3 on 100% of
    #                                                steps, so the clamp returns exactly 0.0 on
    #                                                every step at initialization
    # and the inference that identical zero reward everywhere leaves PPO no advantage variance to
    # bootstrap from. The measurement was right; the inference was wrong. v10 then produced exactly
    # the failure the legged_gym comment warns about: corr(reward, episode_length) = -0.701, and a
    # final policy that survived 23.7 steps versus 33.2 for an UNTRAINED network -- it learned that
    # dying early was cheaper than living.
    #
    # Why the clamp works despite the all-zero start: with reward >= 0 always, a longer episode can
    # never score worse than a shorter one, so early termination is never an improvement. The
    # gradient does not have to come from the clamped random regime -- it appears as soon as any
    # rollout stumbles into a state scoring above 0, and from there the ratchet only goes one way.
    # Bootstrapping out of an all-zero region is a slow start; a suicide gradient is a wrong answer.
    only_positive_rewards: bool = False

    command_lin_vel_range: tuple[float, float] = (-1.0, 1.0)
    command_ang_vel_range: tuple[float, float] = (-1.0, 1.0)
    domain_rand: DomainRandCfg = field(default_factory=DomainRandCfg)
    reward_scales: RewardScales = field(default_factory=RewardScales)

    @property
    def substeps(self) -> int:
        return max(1, round(self.control_dt / self.sim_dt))

    @property
    def max_episode_length_steps(self) -> int:
        return max(1, round(self.max_episode_length_s / self.control_dt))
