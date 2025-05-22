# Test cases 

Three test cases are available : 

* ellipticalwing : Compares the aerodynamic forces over an elliptical wing using the vortex code with a theoretical value.
* newMexico : Compares aerodynamic forces over the newMexico wind turbine blades with results obtained from the CASTOR FVW solver.
* VAWT : 

For each example, it is possible to generate Tecplot (.tp) files, which can be visualized using a post-processing tool to analyze the behavior of the vortex wake behind the studied structure.

# How to 

Include the python folders to be able to run the examples. This can be done before calling the scripts :

``` export PYTHONPATH=/path/to/repo:$PYTHONPATH ```  


Or inside the scripts : 

```python
import sys
sys.path.append('/path/to/repo')
