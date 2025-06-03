import torch
import torch.nn as nn
from pytorch_tcn import TCN


class AutoregressiveTCNModel(nn.Module):
    """
    TCN model for autoregressive residual prediction.
    Takes static features + sequence of previous wait_times/residuals.
    """
    def __init__(self, static_features_size, seq_length, output_size, 
                 num_channels, kernel_size, dropout=0.2, num_layers=8):
        super(AutoregressiveTCNModel, self).__init__()
        
        self.static_features_size = static_features_size
        self.seq_length = seq_length
        
        # The input to TCN will be: static features repeated + autoregressive sequence
        # Autoregressive sequence: seq_length steps of [wait_time, residual] = seq_length * 2
        self.autoregressive_size = seq_length * 2  # wait_time + residual per timestep
        self.total_input_size = static_features_size + self.autoregressive_size
        
        # TCN processes the combined features as a sequence
        # We'll reshape the input to treat it as a sequence
        self.tcn_input_size = static_features_size + 2  # static + [wait_time, residual] per step
        
        channels = [num_channels] * num_layers
        
        self.tcn = TCN(
            num_inputs=self.tcn_input_size,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=True,
            use_skip_connections=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(num_channels, num_channels // 2)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(num_channels // 2, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Split input into static and autoregressive parts
        static_features = x[:, :self.static_features_size]  # (batch_size, static_features)
        autoregressive_features = x[:, self.static_features_size:]  # (batch_size, seq_length * 2)
        
        # Reshape autoregressive features
        autoregressive_reshaped = autoregressive_features.view(
            batch_size, self.seq_length, 2
        )  # (batch_size, seq_length, 2)
        
        # Repeat static features for each timestep
        static_repeated = static_features.unsqueeze(1).repeat(
            1, self.seq_length, 1
        )  # (batch_size, seq_length, static_features_size)
        
        # Combine static and autoregressive features
        combined_sequence = torch.cat([
            static_repeated, autoregressive_reshaped
        ], dim=2)  # (batch_size, seq_length, static_features_size + 2)
        
        # TCN expects (batch_size, features, seq_length)
        tcn_input = combined_sequence.transpose(1, 2)
        
        # TCN forward pass
        tcn_out = self.tcn(tcn_input)  # (batch_size, channels, seq_length)
        
        # Take the last time step
        last_hidden = tcn_out[:, :, -1]  # (batch_size, channels)
        
        # Output layers
        out = self.dropout(last_hidden)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        
        return out
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration dictionary"""
        return cls(
            static_features_size=config['static_features_size'],
            seq_length=config['seq_length'],
            output_size=config.get('output_size', 1),
            num_channels=config['num_channels'],
            kernel_size=config['kernel_size'],
            dropout=config.get('dropout', 0.2),
            num_layers=config.get('num_layers', 8)
        )
    
    def get_config(self):
        """Get model configuration"""
        return {
            'static_features_size': self.static_features_size,
            'seq_length': self.seq_length,
            'autoregressive_size': self.autoregressive_size,
            'total_input_size': self.total_input_size,
            'tcn_input_size': self.tcn_input_size,
            'num_channels': self.tcn.num_channels[0] if hasattr(self.tcn, 'num_channels') else None,
            'kernel_size': self.tcn.kernel_size if hasattr(self.tcn, 'kernel_size') else None,
            'dropout': self.dropout.p,
            'num_layers': len(self.tcn.num_channels) if hasattr(self.tcn, 'num_channels') else None,
            'output_size': self.linear2.out_features
        }