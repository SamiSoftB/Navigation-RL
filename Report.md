[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

[image2]: training_scores.png "Training scores"

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

[image2]: training_scores.png "Training scores"
## The environment

We have a navigation problem described by a continuous space of size 37 and discrete action space with 4 actions. 

## The agent

I use a DQN agent with neural network model: 3 layers with Relu activation functions.


```python:
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

```

## hyperparameters

| hyperparameter              | Value  |
|-----------------------------|--------|
| batch size                  | 64     |
| buffer size                 | 100000 |
| gamma, discount factor      | 0.99   |
| tau, for soft update        | 0.001  |
| learning rate               | 0.0005 |
| update agent every          | 4      |
| initial exploration epsilon | 1      |
| end exploration epsilon     | 0.01   |
| epsilon decay               | 0.995  |
|



## Training scores:


![training][image2]


## Next steps:

I will try to train the agent on pixels. So far I was unable to run the code in a Google Colab in order to use a more powerful GPU.
I have an error : Unity environment waited too long to respond. 

I used the No_vis environment. 
