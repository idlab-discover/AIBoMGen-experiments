# AIBoMGen Research Experiments

This repository contains evaluation scripts and results for the [AIBoMGen](https://github.com/idlab-discover/AIBoMGen) proof of concept repository.

## AIBoMGen Ecosystem

This repository is part of the broader AIBoMGen ecosystem for generating, analyzing, and validating AI/ML Bills of Materials (AIBOMs).

| Repository | Purpose |
|---|---|
| [AIBoMGen CLI](https://github.com/idlab-discover/aibomgen-cli) | Command-line tool for generating AIBOMs from source code and ML artifacts |
| [AIBoMGen CLI Action](https://github.com/CRA-tools/AIBoMGen-cli-action) | GitHub Action for automated AIBOM generation in CI/CD pipelines |
| [AIBoMGen CLI Dashboard](https://github.com/CRA-tools/aibomgen-cli-dashboard) | Demo dashboard using [AIBoMGen CLI](https://github.com/idlab-discover/aibomgen-cli) |
| [AIBoMGen](https://github.com/idlab-discover/AIBoMGen) | Proof of concept research repository |
| [AIBoMGen Experiments](https://github.com/idlab-discover/AIBoMGen-experiments) | Experimental evaluations of [AIBoMGen](https://github.com/idlab-discover/AIBoMGen)|

## How to Run the Experiments

1. Ensure the backend API is running on `localhost:8000` without authentication.
2. Navigate to the `scripts` folder.
3. Run the Jupyter notebooks in the respective subfolders to execute the experiments:
   - `basic_eval/`
   - `performance/`
   - `storage/`

## Results

- The resulting jobs and outputs will be written to the `results` folder.
- Each experiment will create a subfolder in `results` corresponding to its type and configuration.

## Publishing

This repository is published on [Zenodo](https://zenodo.org/records/15505280).

## Contact

For inquiries, feel free to reach out

Maintained by:

Wiebe Vandendriessche  
[wiebe.vandendriessche@ugent.be](mailto:wiebe.vandendriessche@ugent.be)  
[LinkedIn](https://www.linkedin.com/in/wiebe-vandendriessche/?locale=en_US)  
[DISCOVER: IDLab, Ghent University – imec](https://idlab.ugent.be/research-teams/discover).

## License

This project is licensed under the terms described in the [LICENSE](./LICENSE) file.

## Acknowledgements

This work has been partially supported by the [CRACY project](https://cra-cy.eu/), funded by the European Union’s Digital Europe Programme under grant agreement No 101190492.
