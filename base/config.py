import os
import copy
import json

WEIGHTS_NAME = "pytorch_model.bin"
CONFIG_NAME = "config.json"

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                raise err(f"Can't set {key} with value {value} for {self}")

    @classmethod
    def from_json_file(cls, json_file:str):
        """
        Hugging Face의 경우 pretrained_model_name_or_path가 
        local에 존재하는 json 파일이면 해당 파일을 불러오고,
        아니면 url을 통해서 다운로드하는 형식으로 이루어짐

        본 코드에서는 local에서 json 파일을 가져오는 형식만 고려함
        """

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(json_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(
                f"It looks like the config file at '{json_file}' is not a valid JSON file."
            )
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file:str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
    
    def save_pretrained(self, save_directory: str):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                <Tip warning={true}>

                Using `push_to_hub=True` will synchronize the repository you are pushing to with `save_directory`,
                which requires `save_directory` to be a local clone of the repo you are pushing to if it's an existing
                folder. Pass along `temp_dir=True` to use a temporary directory instead.

                </Tip>

            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        self.to_json_file(output_config_file)
    
    def to_json_file(self, json_file_path:str):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
    
    def to_json_string(self):
        output = copy.deepcopy(self.__dict__)
        return json.dumps(output, indent=2, sort_keys=True) + "\n"