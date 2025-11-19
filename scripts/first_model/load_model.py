#%% packages
import torch

#%% model class
class DeepRegression(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(DeepRegression, self).__init__()
        self.linear_in = torch.nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear_out = torch.nn.Linear(in_features=hidden_size, out_features=output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.linear_out(x)
        x = self.relu(x)
        return x
# %% erstelle Modellinstanz
model = DeepRegression(input_size=30, output_size=1, hidden_size=4)

#%% lade Modellgewichte ins Modell
state_dict = torch.load(f="model001.pt")
model.load_state_dict(state_dict=state_dict)
