import requests
import yaml
import click
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class APIConfig:
    address: str
    port: Optional[int] = None


@dataclass
class ModelConfig:
    model_name: str
    hf_token: Optional[str] = None
    args: Optional[str] = None


@dataclass
class ServiceConfig:
    ports: int
    readiness_probe: str


@dataclass
class ResourceConfig:
    cpus: int
    memory: int
    ports: int
    accelerators: str
    image_id: str


class ConfigReader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file."""
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)

                if config is None:
                    raise click.ClickException(
                        f"Configuration file is empty: {self.config_path}"
                    )

                if not isinstance(config, dict):
                    raise click.ClickException(
                        f"Invalid YAML format in {self.config_path}. Expected a dictionary."
                    )

                self._validate_config(config)
                return config
        except FileNotFoundError:
            raise click.ClickException(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise click.ClickException(f"Error parsing YAML file: {e}")

    def _validate_section_exists(self, config: Dict[str, Any], section: str) -> None:
        """Validate that a section exists in the config."""
        if section not in config:
            raise click.ClickException(f"Missing required section: {section}")

    def _validate_field_exists_and_non_empty(
        self, section: Dict[str, Any], field: str, section_name: str
    ) -> None:
        """Validate that a field exists in a section and has a non-empty value."""
        if field not in section or section[field] is None:
            raise click.ClickException(
                f"Missing required field '{field}' in {section_name} section"
            )
        if isinstance(section[field], str) and not section[field].strip():
            raise click.ClickException(
                f"Empty value for required field '{field}' in {section_name} section"
            )

    def _validate_api_section(self, config: Dict[str, Any]) -> None:
        """Validate the API section of the configuration."""
        self._validate_section_exists(config, "api")
        api_section = config["api"]

        if not isinstance(api_section, dict):
            raise click.ClickException("API section must be a dictionary")

        self._validate_field_exists_and_non_empty(api_section, "address", "api")

        url = f"http://{api_section['address']}"
        if api_section["port"]:
            url += f":{api_section['port']}"
        url += "/health"

        try:
            response = requests.get(url)
            if response.status_code != 200:
                pass
                raise click.ClickException(f"Status code is {response.status_code}")
        except requests.RequestException as e:
            pass
            raise click.ClickException(f"API Health Check failed: {e}")

    def _validate_model_section(self, config: Dict[str, Any]) -> None:
        """Validate the Model section of the configuration."""
        self._validate_section_exists(config, "model")
        model_section = config["model"]

        if not isinstance(model_section, dict):
            raise click.ClickException("Model section must be a dictionary")

        self._validate_field_exists_and_non_empty(model_section, "model_name", "model")

        url = f"https://huggingface.co/{model_section['model_name']}"

        try:
            response = requests.get(url)
            if response.status_code != 200:
                raise requests.RequestException(
                    f"Resource not found! Sorry, we can't find the model '{model_section['model_name']}' you are looking for."
                )
        except requests.RequestException as e:
            raise click.ClickException(f"Model Check failed: {e}")

    def _validate_resources_section(self, config: Dict[str, Any]) -> None:
        """Validate the Resources section of the configuration."""
        self._validate_section_exists(config, "resources")
        resources_section = config["resources"]

        if not isinstance(resources_section, dict):
            raise click.ClickException("Resources section must be a dictionary")

        required_fields = ["cpus", "memory", "ports", "accelerators", "image_id"]
        for field in required_fields:
            self._validate_field_exists_and_non_empty(
                resources_section, field, "resources"
            )

        self._validate_docker_image(resources_section["image_id"])

        try:
            _, number = resources_section["accelerators"].split(":")
            int(number)
        except ValueError as e:
            raise click.ClickException(
                "Number of accelerators must be an integer & Accelerator field should be in the format of e.g. MI200:2"
            ) from e
        except Exception as e:
            raise click.ClickException(f"Error validating resources field: {e}")

    def _validate_service_section(self, config: Dict[str, Any]) -> None:
        """Validate the Service section of the configuration."""
        self._validate_section_exists(config, "service")
        service_section = config["service"]

        if not isinstance(service_section, dict):
            raise click.ClickException("Service section must be a dictionary")

        required_fields = ["ports", "readiness_probe"]
        for field in required_fields:
            self._validate_field_exists_and_non_empty(service_section, field, "service")

        if service_section["ports"] != config["resources"]["ports"]:
            raise click.ClickException(
                f"Service port doesn't not match Resources port value: {service_section['ports']}:{config['resources']['ports']}"
            )

    def _validate_docker_image(self, image_id: str) -> None:
        """Validate that a Docker image exists in Docker Hub."""
        if not image_id.startswith("docker:"):
            raise click.ClickException(
                f"Invalid Docker image format. Expected 'docker:repository/image:tag', got: {image_id}"
            )

        # Remove 'docker:' prefix
        image_path = image_id[7:]

        # Split into repository/image and tag
        if ":" not in image_path:
            raise click.ClickException(
                f"Invalid Docker image format. Missing tag in: {image_id}"
            )

        repo, tag = image_path.split(":", 1)

        # Docker Hub API endpoint
        api_url = f"https://hub.docker.com/v2/repositories/{repo}/tags/{tag}/"

        try:
            response = requests.get(api_url)
            if response.status_code == 404:
                raise click.ClickException(f"Docker image not found: {image_id}")
            elif response.status_code != 200:
                raise click.ClickException(
                    f"Failed to check Docker image (status {response.status_code}): {image_id}"
                )
        except requests.RequestException as e:
            raise click.ClickException(f"Error checking Docker image existence: {e}")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the complete configuration structure."""
        self._validate_api_section(config)
        self._validate_model_section(config)
        self._validate_resources_section(config)
        self._validate_service_section(config)

    @property
    def api_config(self) -> APIConfig:
        """Get API configuration."""
        api = self.config["api"]
        return APIConfig(
            address=api["address"],
            port=api.get("port"),  # Make port optional
        )

    @property
    def model_config(self) -> ModelConfig:
        """Get model configuration."""
        model = self.config["model"]
        return ModelConfig(
            model_name=model["model_name"],
            hf_token=model.get("hf_token"),
            args=model.get("args"),  # Using get() to make hf_token optional
        )

    @property
    def resources_config(self) -> ResourceConfig:
        """Get resources configuration."""
        res = self.config["resources"]
        return ResourceConfig(
            cpus=res["cpus"],
            memory=res["memory"],
            ports=res["ports"],
            accelerators=res["accelerators"],
            image_id=res["image_id"],
        )

    @property
    def service_config(self) -> ServiceConfig:
        """Get service configuration."""
        svc = self.config["service"]
        return ServiceConfig(ports=svc["ports"], readiness_probe=svc["readiness_probe"])


@click.group()
def cli():
    """CLI tool to manage configuration."""
    pass


@cli.command()
@click.option(
    "-f",
    "--file",
    type=click.Path(exists=True),
    required=True,
    help="Path to YAML configuration file",
)
def validate(file):
    """Validate the configuration file."""
    try:
        config_reader = ConfigReader(file)

        # API Configuration
        api = config_reader.api_config
        click.echo("API Configuration:")
        click.echo(f"  Address: {api.address}")
        click.echo(f"  Port: {api.port}")

        # Model Configuration
        model = config_reader.model_config
        click.echo("\nModel Configuration:")
        click.echo(f"  Model Name: {model.model_name}")
        click.echo(f"  Huggingface Token: {model.hf_token}")
        click.echo(f"  Huggingface Token: {model.args}")

        # Resources Configuration
        res = config_reader.resources_config
        click.echo("\nResources Configuration:")
        click.echo(f"  CPUs: {res.cpus}")
        click.echo(f"  Memory: {res.memory}")
        click.echo(f"  Ports: {res.ports}")
        click.echo(f"  Accelerators: {res.accelerators}")
        click.echo(f"  Image ID: {res.image_id}")

        # Service Configuration
        svc = config_reader.service_config
        click.echo("\nService Configuration:")
        click.echo(f"  Ports: {svc.ports}")
        click.echo(f"  Readiness Probe: {svc.readiness_probe}")

        click.echo("Configuration is valid!")
    except click.ClickException as e:
        click.echo(f"Configuration error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option(
    "-f",
    "--file",
    type=click.Path(exists=True),
    required=True,
    help="Path to YAML configuration file",
)
def deploy(file):
    """Send the configuration to the API server."""
    try:
        config_reader = ConfigReader(file)
        # Convert the configuration to a JSON-compatible dictionary
        config_data = config_reader.config

        api_url = f"http://{config_reader.config['api']['address']}"
        if config_reader.config["api"]["port"]:
            api_url += f":{config_reader.config['api']['port']}"
        api_url += "/deploy"

        # Send the configuration to the API server
        response = requests.post(api_url, json=config_data)

        # Check the response status
        if response.status_code == 200:
            click.echo(
                "Configuration successfully sent to the API server.\n"
                + str(response.json())
            )
        else:
            click.echo(
                f"Failed to send configuration. Status code: {response.status_code}"
            )
            click.echo(f"Response: {response.text}")

    except click.ClickException as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except requests.RequestException as e:
        click.echo(f"Failed to send configuration: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
