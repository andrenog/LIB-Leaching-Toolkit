# LIB Leaching Toolkit

[![DOI](https://zenodo.org/badge/872375933.svg)](https://doi.org/10.5281/zenodo.16096943)

![LIB Leaching Toolkit Logo](data/icon.png)

The LIB Leaching Toolkit is a Python-based application that helps researchers and engineers evaluate the environmental and economic impacts of lithium-ion battery cathode leaching. This toolkit provides tools to:

* Estimate leaching yields
* Estimate reagent costs
* Assess environmental impacts
* Calculate selectivities
* Calculate enrichment factors
* Compare reaction conditions

... enabling you to make informed decisions.

**Disclaimer:** This toolkit is a work in progress. While we have taken care in its development, there might be bugs or limitations. We appreciate your feedback and understanding as we continue to improve it.

## Installation

The LIB Leaching Toolkit can be installed in one of two ways.

**1. Using Pre-compiled Executables**

This method provides ready-to-run executables for Windows and Linux.

  * Download the appropriate executable for your operating system from the **Releases** page.
  * No installation is required. Simply run the downloaded file.

**2. From Source Code**

This method allows you to access and modify the source code.

  * Set up a virtual environment:
      * **Windows:** `py -3.12 -m venv venv`
      * **macOS/Linux:** `python3.12 -m venv venv`
  * Activate the environment:
      * **Windows:** `venv\Scripts\activate`
      * **macOS/Linux:** `source venv/bin/activate`
  * Install dependencies: `pip install -r requirements.txt`
  * Run the application: `python LIBtoolkit.py`

## Usage

The LIB Leaching Toolkit provides a graphical user interface (GUI) for easy interaction.

1.  **Launch the application:**

      * If using a pre-compiled executable, run the file.
      * If installed from source, run the script: `python LIBtoolkit.py`

2.  **Import your data:**

      * Click the "Browse" button in the "Analysis" tab.
      * Select the Excel file containing your reaction conditions.
      * Ensure your data follows the format provided in the `data/template.xlsx` file.

3.  **View and save results:**

      * Upon importing data, the toolkit automatically calculates predictions and analyses.
      * Plots are saved as PNG files in the `outputs/figures` folder.
      * Excel files with detailed results are saved in the `outputs` folder.

4.  **Explore the plots:**

      * Click the buttons in the "Analysis" tab to generate and view plots for yields, selectivity, costs, and environmental impacts.

### Examples and Templates

An example template file, showing the required format for your input data, is included with this software. You can find it in the `data` folder:

* `data/template.xlsx`

## Input Generation

The toolkit can generate input tables for systematically studying reaction conditions.

1.  Navigate to the **"Input generation"** tab.
2.  **Select the condition to vary** from the dropdown menu.
3.  **Set the range and steps** for the selected condition.
4.  **Enter fixed values** for all other conditions.
5.  Click **"Generate table"**. The table is saved to the `outputs` folder and the analysis is automatically run and displayed in the **"Analysis"** tab.

## Data Files

The `data` folder contains Excel files that store the default values for costs (`Acids_costs.xlsx`) and environmental impact multipliers (`Acids_impacts.xlsx`). You can modify these files to customize the data used by the toolkit.
<!-- 
**Important Note:** The impact multipliers are placeholders and **should be updated by the user** with values obtained from their preferred Life Cycle Assessment (LCA) software. The LCA software used to generate the initial values does not permit redistribution of this data.
 -->


## Requirements

  * **Python:** This project requires Python 3.12 specifically. It will not work with Python 3.13 or newer due to library compatibility constraints with the pre-trained machine learning model.
  * **Operating System:**
      * The source code can be run on Windows, macOS, or Linux.
      * Pre-compiled executables for Windows are provided.
  * **Dependencies:** All required Python packages are listed in the `requirements.txt` file and are installed when following the source code installation instructions.

## License

This project is licensed under the MIT License - see the [License](LICENSE) file for details.

## Citation

If you use this software in your research, please cite it using the DOI https://doi.org/10.5281/zenodo.16096943


## Disclaimer

This software is provided "as is" without warranty of any kind, express or implied, including but not to limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

## Acknowledgments
