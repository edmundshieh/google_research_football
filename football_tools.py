import os
from typing import TYPE_CHECKING, Dict, Optional

import gfootball
import math
import gym
import numpy as np
from gfootball import env as fe
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID

ROLES = {
    0: 'GK',
    1: 'CB',
    2: 'LB',
    3: 'RB',
    4: 'DM',
    5: 'CM',
    6: 'LM',
    7: 'RM',
    8: 'AM',
    9: 'CF'
}

ACTIONS = {
    0: 'idle',
    1: 'left',
    2: 'top_left',
    3: 'top',
    4: 'top_right',
    5: 'right',
    6: 'bottom_right',
    7: 'bottom',
    8: 'bottom_left',
    9: 'long_pass',
    10: 'high_pass',
    11: 'short_pass',
    12: 'shot',
    13: 'sprint',
    14: 'release_direction',
    15: 'release_sprint',
    16: 'sliding',
    17: 'dribble',
    18: 'release_dribble',
    19: 'builtin_ai',
}

STICKY_ACTIONS = {
    0: 'left',
    1: 'top_left',
    2: 'top',
    3: 'top_right',
    4: 'right',
    5: 'bottom_right',
    6: 'bottom',
    7: 'bottom_left',
    8: 'sprint',
    9: 'dribble'
}

GAME_MODE = {
    0: 'Normal',
    1: 'KickOff',
    2: 'GoalKick',
    3: 'FreeKick',
    4: 'Corner',
    5: 'ThrowIn',
    6: 'Penalty'
}


def get_obs_act_space(env_name):
    env = RllibGFootball(env_name=env_name)
    obs_space = env.observation_space
    act_space = env.action_space
    del env
    return obs_space, act_space

def env_name_to_n_players(env_name):
    n_players = int(env_name[0])
    if 'auto_GK' in env_name:
        n_players -= 1
    return n_players

def n_players_to_env_name(n_players, auto_GK):
    env_name = f"{n_players}_vs_{n_players}"
    GK_addon = "_auto_GK" if auto_GK else ""
    env_name += GK_addon
    return env_name

def create_football_env(env_name, n_controls, write_video, render, logdir, seed=None):
    gfootball_dir = os.path.dirname(gfootball.__file__)
    assert os.path.exists(gfootball_dir), "Couldn't find gfootball package, make sure it is installed"
    scenarios_dir = os.path.join(gfootball_dir, "scenarios")
    assert os.path.exists(scenarios_dir), "Couldn't find gfootball scenarios folder, make sure it is installed"

    scenario_file_name = f"{env_name}.py"
    scenarios_gfootbal_file = os.path.join(scenarios_dir, scenario_file_name)
    if not os.path.exists(scenarios_gfootbal_file):
        local_dir = os.path.dirname(__file__)
        local_scenario_file = os.path.join(local_dir, '..', '..', 'docker', scenario_file_name)
        assert os.path.exists(local_scenario_file), f"Couldn't find {local_scenario_file}, can't copy it to {scenarios_dir}"
        from shutil import copyfile
        copyfile(local_scenario_file, scenarios_gfootbal_file)

    assert os.path.exists(scenarios_gfootbal_file), f"Couldn't find {scenarios_gfootbal_file}, make sure you manually copy {scenario_file_name} to {scenarios_dir}"

    env = fe.create_environment(
        env_name=env_name,
        stacked=False,
        representation='simple115v2',
        # scoring is 1 for scoring a goal, -1 the opponent scoring a goal
        # checkpoint is +0.1 first time player gets to an area (10 checkpoint total, +1 reward max)
        rewards='checkpoints,scoring',
        logdir=logdir,
        write_goal_dumps=write_video,
        write_full_episode_dumps=write_video,
        render=render,
        write_video=write_video,
        dump_frequency=1 if write_video else 0,
        extra_players=None,
        number_of_left_players_agent_controls=n_controls,
        number_of_right_players_agent_controls=0, 
        other_config_options = {'game_engine_random_seed': seed} if seed is not None else {})
    
    env = CustomRewardWrapper(env)

    return env


class RllibGFootball(MultiAgentEnv):
    EXTRA_OBS_IDXS = np.r_[6:22,28:44,50:66,72:88,100:108]

    def __init__(self, env_name, write_video=False, render=False, logdir='/tmp/football', seed=None):
        self.n_players = env_name_to_n_players(env_name)
        self.env = create_football_env(env_name, self.n_players, write_video, render, logdir, seed=seed)

        self.action_space, self.observation_space = {}, {}
        for idx in range(self.n_players):
            self.action_space[f'player_{idx}'] = gym.spaces.Discrete(self.env.action_space.nvec[idx]) \
                if self.n_players > 1 else self.env.action_space
            lows = np.delete(self.env.observation_space.low[idx], RllibGFootball.EXTRA_OBS_IDXS)
            highs = np.delete(self.env.observation_space.high[idx], RllibGFootball.EXTRA_OBS_IDXS)
            self.observation_space[f'player_{idx}'] = gym.spaces.Box(
                low=lows, high=highs, dtype=self.env.observation_space.dtype) \
                if self.n_players > 1 else self.env.observation_space

        self.reward_range = np.array((-np.inf, np.inf))
        self.metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
        self.spec = None

    def _tidy_obs(self, obs):
        for key, values in obs.items():
            obs[key] = np.delete(values, RllibGFootball.EXTRA_OBS_IDXS)
        return obs

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for idx in range(self.n_players):
            obs[f'player_{idx}'] = original_obs[idx] \
                if self.n_players > 1 else original_obs
        return self._tidy_obs(obs)

    def step(self, action_dict):

        actions = []
        for idx in range(self.n_players):
            actions.append(action_dict[f'player_{idx}'])
        o, r, d, i = self.env.step(actions)

        game_info = {}
        for k, v in self.env.unwrapped._env._observation.items():
            game_info[k] = v

        scenario = self.env.unwrapped._env._config['level']
        obs, rewards, dones, infos = {}, {}, {}, {}
        for idx in range(self.n_players):
            obs[f'player_{idx}'] = o[idx] \
                if self.n_players > 1 else o
            rewards[f'player_{idx}'] = r[idx] \
                if self.n_players > 1 else r
            dones[f'player_{idx}'] = d
            dones['__all__'] = d
            infos[f'player_{idx}'] = i
            infos[f'player_{idx}']['game_scenario'] = scenario
            infos[f'player_{idx}']['game_info'] = game_info
            infos[f'player_{idx}']['action'] = action_dict[f'player_{idx}']

        return self._tidy_obs(obs), rewards, dones, infos


class FootballCallbacks(DefaultCallbacks):
    STAT_FUNC = {
        'mean': np.mean
    }

    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        episode: MultiAgentEpisode,
                        env_index: Optional[int] = None,
                        **kwargs) -> None:
        # Only operate on every 10th episode
        if episode.episode_id % 10 != 0:
            return
        
        # pull agent specific data
        for agent_id in episode.get_agents():
            # get the agent id
            a_idx = int(agent_id.split("_")[-1])

            # get agent observations into tensorboard
            obs = episode.last_observation_for(agent_id)
            for obs_idx, obs_value in enumerate(obs):
                key = f"agent_{a_idx}/observation_{obs_idx}"
                val = obs_value
                episode.user_data.setdefault(key, []).append(val)

            # pull last info from episode
            info = episode.last_info_for(agent_id)

            # get agent action selections into tensorboard
            for action_id, name in ACTIONS.items():
                key = f"agent_{a_idx}/action_{name}_selected"
                val = int(action_id == info['action'])
                episode.user_data.setdefault(key, []).append(val)



        # pull simulation info, info is the same for all agents
        # for meaning check link below
        # https://github.com/google-research/football/blob/v2.10.2/gfootball/doc/observation.md
        info = info['game_info']

        # - `ball_owned_team` - {-1, 0, 1}, -1 = ball not owned, 0 = left team, 1 = right team.
        none_owns = info['ball_owned_team'] == -1
        left_owns = info['ball_owned_team'] == 0
        right_owns = info['ball_owned_team'] == 1
        episode.user_data.setdefault("ball/nobody_owns", []).append(int(none_owns))
        episode.user_data.setdefault("ball/left_team_owns", []).append(int(left_owns))
        episode.user_data.setdefault("ball/right_team_owns", []).append(int(right_owns))

        # - `ball_owned_player` - {0..N-1} integer denoting index of the player owning the ball.
        zero_owns = info['ball_owned_player'] == 0
        one_owns = info['ball_owned_player'] == 1
        two_owns = info['ball_owned_player'] == 2
        episode.user_data.setdefault("ball/left_team_player_0_owns", []).append(
            int(left_owns and zero_owns))
        episode.user_data.setdefault("ball/left_team_player_1_owns", []).append(
            int(left_owns and one_owns))
        episode.user_data.setdefault("ball/left_team_player_2_owns", []).append(
            int(left_owns and two_owns))
        episode.user_data.setdefault("ball/right_team_player_0_owns", []).append(
            int(right_owns and zero_owns))
        episode.user_data.setdefault("ball/right_team_player_1_owns", []).append(
            int(right_owns and one_owns))
        episode.user_data.setdefault("ball/right_team_player_2_owns", []).append(
            int(right_owns and two_owns))

        # all teams info
        for team in ['left_team']: # Only care about our team
            # all players in the `team`
            positions = []

            for player_idx in range(len(info[team])):
                # player position
                position = []
                for position_idx, name in zip(range(len(info[team][player_idx])), ['x', 'y']):
                    key = f"{team}/player_{player_idx}/position_{name}"
                    val = info[team][player_idx][position_idx]
                    episode.user_data.setdefault(key, []).append(val)
                    position.append(val)
                positions.append(position)

                # # player is agent/controllable
                # key = f"{team}/player_{player_idx}/is_controllable"
                # ikey = f'{team}_controlled_player'.replace('team', 'agent')
                # control_players_idx = info[ikey]
                # val = int(player_idx in control_players_idx)
                # episode.user_data.setdefault(key, []).append(val)
                
                # # sticky actions
                # for action_id, name in STICKY_ACTIONS.items():
                #     key = f"{team}/player_{player_idx}/sticky_action_{name}"
                #     ikey = f'{team}_sticky_actions'.replace('team', 'agent')
                #     control_player_idx = control_players_idx.index(player_idx) \
                #         if player_idx in control_players_idx else None
                #     val = 0 if control_player_idx is None else info[ikey][control_player_idx][action_id]
                #     episode.user_data.setdefault(key, []).append(val)
            
            # Calculate distances between controllable players
            distance = math.sqrt(((positions[2][0]-positions[1][0])*50)**2 + ((positions[2][1]-positions[1][1])*50)**2)
            
            # Increment the counter if the distance is greater than 20m
            key = f"{team}/cooperation/distance"
            episode.user_data.setdefault(key, []).append(distance)

            key = f"{team}/cooperation/distance_greater_than_20_count"
            if distance > 20:
                episode.user_data.setdefault(key, []).append(1)
            else:
                episode.user_data.setdefault(key, []).append(0)

            # Compare sticky actions 
            # Directions of travel
            going_left = {0, 1, 7}
            going_right = {3, 4, 5}

            ikey = f'{team}_controlled_player'.replace('team', 'agent')
            control_players_idx = info[ikey]
            ikey_player = f'{team}_sticky_actions'.replace('team', 'agent')
            control_player_1_idx = control_players_idx.index(1)
            control_player_2_idx = control_players_idx.index(2)

            # Vectors of sticky actions
            agent_0_sticky = info[ikey_player][control_player_1_idx]
            agent_1_sticky = info[ikey_player][control_player_2_idx]

            key = f"{team}/cooperation/same_sticky_action_count"
            same_sticky_action = any(i == j == 1 for i, j in zip(agent_0_sticky, agent_1_sticky))
            same_going_left = any(agent_0_sticky[i] == agent_1_sticky[i] == 1 for i in going_left)
            same_going_right = any(agent_0_sticky[i] == agent_1_sticky[i] == 1 for i in going_right)

            if same_sticky_action or same_going_left or same_going_right:
                episode.user_data.setdefault(key, []).append(1)
            else:
                episode.user_data.setdefault(key, []).append(0)

            key = f"{team}/cooperation/same_exact_sticky_action_count"
            same_sticky_action = any(i == j == 1 for i, j in zip(agent_0_sticky, agent_1_sticky))
            if same_sticky_action:
                episode.user_data.setdefault(key, []).append(1)
            else:
                episode.user_data.setdefault(key, []).append(0)


    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        # Only operate on every 10th episode
        if episode.episode_id % 10 != 0:
            return

        # get infor from the first agent
        last_info = episode.last_info_for(episode.get_agents()[0])

        # store the episode score reward
        score_reward = last_info['score_reward']
        episode.custom_metrics["game_result/score_reward_episode"] = score_reward

        # game result string and log
        game_result = "loss" if last_info['score_reward'] == -1 else \
            "win" if last_info['score_reward'] == 1 else "tie"
        episode.custom_metrics["game_result/win_episode"] = int(game_result == "win")
        episode.custom_metrics["game_result/tie_episode"] = int(game_result == "tie")
        episode.custom_metrics["game_result/loss_episode"] = int(game_result == "loss")
        episode.custom_metrics["game_result/undefeated_episode"] = int(game_result != "loss")

        # summarize per episode all custom metrics collected on each "step"
        for key, values in episode.user_data.items():
            for fname, func in FootballCallbacks.STAT_FUNC.items():
                episode.custom_metrics[f"{key}_timestep_{fname}_episode"] = func(values)

class FootballCallbacksGroup(DefaultCallbacks):
    STAT_FUNC = {
        'min': np.min,
        'mean': np.mean,
        'max': np.max
    }

    def on_train_result(self, trainer, result):
        if result["training_iteration"] % 10 == 0:
            print("Total training timesteps so far: ", result["timesteps_total"])

class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.previous_ball_owner = None
        self.current_ball_team = None
        self.previous_ball_position = None
        self.furthest_ball_position_left = None 
        self.furthest_ball_position_right = None 
        self.time_penalty = 0.001  # The penalty to apply at each time step
        self.last_actions = None

    def step(self, action):
        self.last_actions = action  # Store the actions
        return super().step(action)

    def reset(self, **kwargs):
        self.previous_ball_owner = None
        self.current_ball_team = None
        self.previous_ball_team = None
        self.previous_ball_position = None
        self.furthest_ball_position_left = 0 
        self.furthest_ball_position_right = 0 
        return self.env.reset(**kwargs)

    def reward(self, reward):
        game_info = self.env.unwrapped._env._observation
        self.current_ball_owner = game_info['ball_owned_player']
        self.current_ball_team = game_info['ball_owned_team']
        self.current_ball_position = game_info['ball']  # (x, y, z) ball position

        # # If the ball has changed possession, reset the furthest_ball_position
        # if self.current_ball_team != self.previous_ball_team:
        #     self.furthest_ball_position_left = self.current_ball_position[0]
        #     self.furthest_ball_position_right = self.current_ball_position[0]

        # if ((self.current_ball_team == self.previous_ball_team) and (self.previous_ball_team not in (None,-1)) and # Team shouldn't change or be -1
        #     (self.current_ball_owner != self.previous_ball_owner)): # test for successful pass
        #     if self.current_ball_team == 0:  # If the current team is the left team
        #         if self.current_ball_position[0] > self.furthest_ball_position_left + 0.1:  # If the ball has moved forward more than 10m
        #             reward += 0.1
        #             self.furthest_ball_position_left = self.current_ball_position[0]  # Update the furthest position
        #     if self.current_ball_team == 1:  # If the current team is the right team
        #         if self.current_ball_position[0] < self.furthest_ball_position_right -0.1:  # If the ball has moved forward more than 10m
        #             reward -= 0.1
        #             self.furthest_ball_position_right = self.current_ball_position[0]  # Update the furthest position


        if self.current_ball_team == 0:
            reward += 0.001
        elif self.current_ball_team == 1:
            reward -= 0.001

        if self.previous_ball_team != self.current_ball_team:
            reward += 0.05 if self.current_ball_team == 0 else -0.05

            # Check if a shot was taken
        if self.last_actions is not None:
            if 12 in self.last_actions: 
            # Check if the ball is near the opponent's goal or your goal
                if (self.current_ball_team == 0 and self.current_ball_position[0] > 0.8 and abs(self.current_ball_position[1])<0.1):
                    reward += 0.1 # Add a reward for a shot on goal
                if (self.current_ball_team == 1 and self.current_ball_position[0] < -0.8 and abs(self.current_ball_position[1])<0.1):
                    reward -= 0.1  

        self.previous_ball_owner = self.current_ball_owner
        self.previous_ball_team = self.current_ball_team
        self.previous_ball_position = self.current_ball_position

        reward -= self.time_penalty  # Apply the time penalty

        return reward
