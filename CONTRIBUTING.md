# CONTRIBUTING

Thank you for your interest in contributing! We welcome contributions of all kinds, including bug reports, feature requests, code improvements and suggestions for documentation.

# How to contribute 

- Make sure all required dependencies are installed and working in your environment.
- Open an issue to briefly describe the feature, bug fix, or improvement you plan to work on.
- Implement and test your changes locally.
- Open a pull request with a small summary of what you’ve done and why. You can tag the issue number if relevant and request a review.

# Code guidelines

No strict style guide is imposed, but it is recommended to follow the existing structure and formatting to keep the codebase consistent. Adding comments to improve clarity is encouraged. If applicable, include tests for any new features or bug fixes.

# Ideas of improvements

Contributions are welcome in all aspects. If you are looking for inspiration, here are some potential directions which could benefit the code :

- Rigid body handling : The `updateTurbine()` function is not generic and doesn't support yaw, tilt, precone or other typical rotor configurations. Making it more flexible would be a valuable improvement.
- Aerolastic coupling : The code could benefit from coupling with a structural solver to enable aeroelastic simulations.