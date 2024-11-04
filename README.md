## LLAMBO-FT: Fine-tuned Large Language Models for Bayesian Optimization

Built on top of [LLAMBO](https://github.com/tennisonliu/LLAMBO/).

## 1. Setup

1. If using OpenAI for original LLAMBO, set up environment variables:

```
echo "export OPENAI_API_KEY={api_key}" >> ~/.zshrc
echo "export OPENAI_API_VERSION={api_version}" >> ~/.zshrc
## Note: these might be optional
echo "export OPENAI_API_BASE={api_base}" >> ~/.zshrc
echo "export OPENAI_API_TYPE={api_type}" >> ~/.zshrc
```
In our experiments, we used ```gpt-turbo-3.5``` for all modules and ```gpt-turbo-3.5-instruct``` for the generative surrogate model (Note: these models might require separate set of credentials).

2. Update the shell with the new variables:
```
source ~/.zshrc
```

3. Confirm that environmental variables are set:
```
echo $OPENAI_API_KEY
echo $OPENAI_API_VERSION
echo $OPENAI_API_BASE
echo $OPENAI_API_TYPE
```

4. Set up Conda environment:
```
git clone https://github.com/tennisonliu/llambo.git
conda create -n llambo python=3.10.0
conda install jupyter
conda activate llambo
## Note: {project_dir} is the path to where to your local directory
export PROJECT_DIR={project_dir}
conda env config vars set PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}
conda env config vars set PROJECT_DIR=${PROJECT_DIR}
conda deactivate
conda activate llambo
```

5. Install requirements:
```
pip install -r requirements.txt
```

---

## 2. Reproducing Results

To generate the training data for finetuning, run:
```bash
./generate_training_data.sh
```