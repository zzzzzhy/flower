import flwr as fl
import requests

def fit_config(server_round: int, address):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 32,
        "current_round": server_round,
        "address": address,
    }
    return config

def main() -> None:
    # Define strategy
    res = requests.get('http://localhost:5924/contract-address')
    if res.get_json() and res.get_json().get('code') == 0:
        credaddress = res.get_json().get('data')
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        on_fit_config_fn=fit_config(address=credaddress)
    )
    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
