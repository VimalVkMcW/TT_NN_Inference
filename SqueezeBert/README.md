# Create a README.md file with the provided content

readme_content = """
# TT_NN Inference

## Project Description
**TT_NN Inference** is a project focused on performing Comparision between the torch model and the local reference model. The project is organized into four primary directories: `datasets`, `reference`, `test`, and `utils`, each serving a distinct purpose in the overall workflow.

## Folder Structure

```bash
SqueezeBert/
├── datasets/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer.json
├── reference/
│   ├── configuration_sequeezebert.py
│   ├── SqueezeBert_Attention.py
│   ├── SqueezeBert_ConvActivation.py
│   └── .....
├── test/
│   ├── Compare_pcc.py
│   ├── test_Squeezebert_Attention.py
│   ├── test_Squeezebert_ConvActivation.py
│   └── .......
├── utils/
│   ├── activation.py
│   ├── generic.py
│   ├── import_utils.py
│   └── .....
├── Requirements.txt
└── README.md
```

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
    To clone a specific branch of this repository, use the following command:
   ```bash
   git clone -b SqueezeBert_b https://github.com/VimalVkMcW/TT_NN_Inference.git

   ```

2. **Navigate to the project directory**:
   ```bash
   cd SqueezeBert
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage


### Running Tests

To execute the tests, use:

```bash
cd test/
pytest 
```
