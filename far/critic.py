"""
Critic networks.
"""
import torch
import torch.nn as nn
from data4robotics.agent import Agent

class Critic(Agent):
    def __init__(
        self,
        features,
        shared_mlp,
        odim,
        n_cams,
        use_obs,
        ac_dim,
        ac_chunk,
        imgs_per_cam=1,
        dropout=0,
        share_cam_features=False,
        feat_batch_norm=True,
        state_only=False,
    ):
        super().__init__(
            features,
            None,
            shared_mlp,
            odim,
            n_cams,
            use_obs,
            imgs_per_cam,
            dropout,
            share_cam_features,
            feat_batch_norm,
            state_only=state_only,
        )
        act_dim = ac_dim * ac_chunk
        self._ac_dim, self._ac_chunk = ac_dim, ac_chunk
        # Preference head (state + action)
        self.pref_head = nn.Sequential(
            nn.Linear(self.obs_enc_dim + act_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        # Value head (state only)
        self.value_head = nn.Sequential(
            nn.Linear(self.obs_enc_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, imgs, obs, ac_flat):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, device = obs.shape[0], obs.device
        s_t = self._shared_forward(imgs, obs)
        actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
        s = torch.cat((s_t, actions.view(B, -1)), dim=-1)
        p = self.pref_head(s)
        
        return p

    def encode_state(self, imgs, obs):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        s_t = self._shared_forward(imgs, obs)
        return s_t
        
    def get_preference(self, imgs, obs, ac_flat):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, device = obs.shape[0], obs.device
        s_t = self._shared_forward(imgs, obs)        
        actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
        s = torch.cat((s_t, actions.view(B, -1)), dim=-1)
        p = self.pref_head(s)
        
        return p

    def get_pref_guidance(self, s_t, ac_flat):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B = s_t.shape[0]
        actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
        s = torch.cat((s_t, actions.view(B, -1)), dim=-1)
        p = self.pref_head(s)
        
        return p

    # Used for evaluating multiple actions per state (TTA)
    def get_sample_preference(self, imgs, obs, ac_flat, num_actions):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, device = obs.shape[0], obs.device
        B = num_actions * B
        s_t = self._shared_forward(imgs, obs)
        s_t = s_t.repeat_interleave(num_actions, dim=0)
        
        actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
        s = torch.cat((s_t, actions.view(B, -1)), dim=-1)
        p = self.pref_head(s)
        
        return p

    def get_value(self, imgs, obs):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, device = obs.shape[0], obs.device
        # with torch.no_grad():
        s_t = self._shared_forward(imgs, obs)

        v = self.value_head(s_t)
        
        return v
    
    def get_value_with_pref(self, imgs, obs, ac_flat):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, device = obs.shape[0], obs.device
        with torch.no_grad():
            s_t = self._shared_forward(imgs, obs)
            actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
            s = torch.cat((s_t, actions.view(B, -1)), dim=-1)
            p = self.pref_head(s)
        
        v = self.value_head(s_t)
        
        return p, v

    @property
    def ac_chunk(self):
        return self._ac_chunk

    @property
    def ac_dim(self):
        return self._ac_dim
    
    
class Critic_2head(Agent):
    def __init__(
        self,
        features,
        shared_mlp,
        odim,
        n_cams,
        use_obs,
        ac_dim,
        ac_chunk,
        imgs_per_cam=1,
        dropout=0,
        share_cam_features=False,
        feat_batch_norm=True,
        state_only=False,
    ):
        super().__init__(
            features,
            None,
            shared_mlp,
            odim,
            n_cams,
            use_obs,
            imgs_per_cam,
            dropout,
            share_cam_features,
            feat_batch_norm,
            state_only=state_only,
        )
        act_dim = ac_dim * ac_chunk
        self._ac_dim, self._ac_chunk = ac_dim, ac_chunk
        # Preference head (state + action)
        self.Q1 = nn.Sequential(
            nn.Linear(self.obs_enc_dim + act_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.Q2 = nn.Sequential(
            nn.Linear(self.obs_enc_dim + act_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        
        # Value head (state only)
        self.value_head = nn.Sequential(
            nn.Linear(self.obs_enc_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, imgs, obs, ac_flat):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, device = obs.shape[0], obs.device
        s_t = self._shared_forward(imgs, obs)
        actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
        s = torch.cat((s_t, actions.view(B, -1)), dim=-1)
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        
        return q1, q2

    def encode_state(self, imgs, obs):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        s_t = self._shared_forward(imgs, obs)
        return s_t
        
    def get_preference(self, imgs, obs, ac_flat):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, device = obs.shape[0], obs.device
        s_t = self._shared_forward(imgs, obs)        
        actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
        s = torch.cat((s_t, actions.view(B, -1)), dim=-1)
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        
        return q1, q2

    def get_pref_guidance(self, s_t, ac_flat):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B = s_t.shape[0]
        actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
        s = torch.cat((s_t, actions.view(B, -1)), dim=-1)
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        
        return q1, q2

    # Used for evaluating multiple actions per state (TTA)
    def get_sample_preference(self, imgs, obs, ac_flat, num_actions):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, device = obs.shape[0], obs.device
        B = num_actions * B
        s_t = self._shared_forward(imgs, obs)
        s_t = s_t.repeat_interleave(num_actions, dim=0)
        
        actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
        s = torch.cat((s_t, actions.view(B, -1)), dim=-1)
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        
        return torch.min(q1, q2)

    def get_value(self, imgs, obs):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, device = obs.shape[0], obs.device
        # with torch.no_grad():
        s_t = self._shared_forward(imgs, obs)

        v = self.value_head(s_t)
        
        return v
    
    def get_value_with_pref(self, imgs, obs, ac_flat):
        """
        cond: dict with key state/rgb; more recent obs at the end
            state: (B, To, Do)
        """
        B, device = obs.shape[0], obs.device
        with torch.no_grad():
            s_t = self._shared_forward(imgs, obs)
            actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
            s = torch.cat((s_t, actions.view(B, -1)), dim=-1)
            q1 = self.Q1(s)
            q2 = self.Q2(s)
        
        v = self.value_head(s_t)
        
        return q1, q2, v

    @property
    def ac_chunk(self):
        return self._ac_chunk

    @property
    def ac_dim(self):
        return self._ac_dim