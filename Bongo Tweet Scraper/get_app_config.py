import yaml

def get_app_config(path = "app_config.yml"):
    with open(path) as f:
        app_config = yaml.load(f)
    return app_config

