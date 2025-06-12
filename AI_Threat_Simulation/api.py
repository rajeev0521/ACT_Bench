from flask import Flask, request, jsonify
import torch
from gan.generator import ThreatGenerator
from stable_baselines3 import PPO

app = Flask(__name__)

# Load generator model
generator = ThreatGenerator(noise_dim=100, output_dim=10)
generator.load_state_dict(torch.load("updated_generator.pth"))
generator.eval()

# Load RL agent
rl_agent = PPO.load("rl_agent")

@app.route("/generate-threat", methods=["POST"])
def generate_threat():
    """API endpoint to generate a threat pattern."""
    request_data = request.get_json()
    noise = torch.tensor(request_data["noise"], dtype=torch.float32).unsqueeze(0)
    threat_pattern = generator(noise).detach().numpy().flatten()

    # RL agent predicts action for the generated threat
    observation = threat_pattern
    action, _states = rl_agent.predict(observation)

    # Map action to a description
    actions_map = {0: "Mitigate", 1: "Ignore", 2: "Log"}
    action_description = actions_map[action]

    return jsonify({
        "threat_pattern": threat_pattern.tolist(),
        "predicted_action": action_description
    })

if __name__ == "__main__":
    app.run(debug=True)
