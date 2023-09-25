import flwr as fl
import requests
credaddress = ''
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "batch_size": 32,
        "current_round": server_round,
        "address": credaddress,
    }
    return config

def main() -> None:
    # Define strategy
    res = requests.get('http://172.17.0.1:5924/contract-address')
    if res.json() and res.json().get('code') == 0:
        credaddress = res.json().get('data')
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        on_fit_config_fn=fit_config
    )
    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
