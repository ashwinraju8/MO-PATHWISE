<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->




<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Multi-Objective Optimization with Path Planning</h3>

  <p align="center">
    Using Multi-Objective Optimization and Splines for Motion Planning
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#methodology-and-psuedoalgorithm">Methodology And Pseudoalgorithm</a>
      <ul>
        <li><a href="#psuedoalgorithm">Psuedoalgorithm</a></li>
        <li><a href="#methodology">Methodology</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


This Git Repository encompasses the final project for Algorithmic Motion Plannning at University of Colorado Boulder during the Fall 2023 semester. The project explores the use of Multi-Objective Optimization (MOO) to perform motion planning. Instead of using traditional optimizers like CHOMP, which maximizes a single objective, path smoothness, while performing obstacle avoidance, 
U(\xi) = f_{\text{smoothness}}(\xi) + f_{\text{obs}}(\xi), the MOO framework looks at multiple objectives while avoiding obstacles. Single objective optimization gives the single best solution for optimizing the criteria, but MOO gives a set of solutions which are all valid and equal in score. 

The utility of MOO is demonstrated when mutiple objectives strive to reach their maximum (or minimum) in a constrained space. For example: say a company is awarded a contract to design a vehicle where the customer wants cost to be minimized, and range and payload to be maximized. Maximizing range may result in a vehicle that cannot carry as much payload. And, conversely, maximizing payload may result in a vehicle that cannot go very far. Or, picking high values for both range and payload could lead to a costly design. All are possible realities, and all could be optimal solutions. MOO will give you a set of "Pareto efficient" solutions, where one solution in the set isn't the most dominant. They are all equal in score. So, it is possible for a vehicle with higher range, lower payload, and low cost to be optimal. And, it's also possible for a vehicle with higher payload, lower range, and low cost to also be optimal. The set of all possible Pareto efficient solutions is called a Pareto Front; one is demonstarted in our results below. 

Originally, we looked at optimizing path distance and and smoothness. However, we observed that the objectives were not very competitive with each other. 

Our first set of objectives to optimize were: 
* path distance 
* path smoothness 

Thus, we chose objectives that were seemingly contentious with eachother:
* path distance (minimize)
* safety buffer (distance to obstacles) 


### Built With

- [NumPy](NumPy-url)
- [Matplotlib](Matplotlib-url)
- [NetworkX](NetworkX-url)
- [Geomdl](Geomdl-url)
- [SciPy](SciPy-url)
- [DEAP](DEAP-url)



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Methodology And Pseudoalgorithm -->
## Methodology & Pseudoalgorithm


### Pseudoalgorithm

Run Evolutionary Algorithm: 
  input parameter: Initial Population (16)
  input parameter: Max Number of Generations (100)
  input parameter: No Improvement Limit (10)
  Part 1:
  1. Set Population equal to the Initial Population 
  2. Initialize Best Fitness Score to zero 
  3. Initialize a counter for No Improvement Count to zero 
  Part 2: 
  1. Assign Best Fitness Score to the Population 
  Part 3: 
  1. Do the following until Max Generation is reach:
  2. Select the best individuals from the population as the Offspring
  3. If the probability of breeding is greater than a  randomly generated probability breed two individuals from the offspring  (child) and replace the parents from the offspring with the two children 
  4. Take two individuals from the offspring and mutate. Replace parents in offspring by the mutated individuals (children)
  5. Delete fitness scores of the changed individuals and calculate new fitness values
  6. Combine the original population with the offspring 
  7. Select the best individuals from the combined population. Only retain the same number of the original population 
  8. Loop through the population to check if the current fitness is better than the previous Best Fitness Score 
  9. If true: redefine Best Fitness Score to the current fitness score and set No Improvement Limit to zero
  10. If False: count +1 to the No Improvement Limit
  11. IF the No Improvement Limit Count > No Improvement Limit, break the loop 

### Methodology

Understanding data structures is crucial to understanding the evolutionary algorithm. Firstly, let's define the structure of the input parameter: Initial Population. 

Initial Population: Starts with 16 individuals, where a an individual is defined as: 

\[ I = \{z, \sigma\} \]

The individuals are comprised of design variables, controls points and weights, and sigmas (step sizes), each of which correspond to the design variables. So, the dimension of one individual is 1 x 6 x number of controls points. It is the sigmas that are being mutated or (sometimes) breeded at every generation of a new population. 

\[ z = \begin{bmatrix}
w_0 & x_1 & y_1 & w_1 & \ldots & x_{n-1} & y_{n-1} & w_{n-1} & w_n
\end{bmatrix} \]

\[ \sigma = \begin{bmatrix}
\sigma_{x0} & \sigma_{y0} & \sigma_{w0} & \sigma_{x1} & \ldots & \sigma_{x(n-1)} & \sigma_{y(n-1)} & \sigma_{w(n-1)} & w_n
\end{bmatrix} \]

<p>The Breeding operation averages the step sizes of two randomly selected individuals \(I_{i,t}\) and \(I_{j,t}\) at generation \(t\):</p>

\[ \sigma_{i,t+1} = \sigma_{j,t+1} = 0.5 (\sigma_{i,t} + \sigma_{j,t}) \]

This operation occurs only if the probability of breeding exceeds a randomly generated probabiity. 

<p>The Mutation operation occurs at every generation of a new population. The mutation of the sigmas of two randomly selected individuals \(I_{i,t}\) and \(I_{j,t}\) at generation \(t\):</p>

\[ \sigma_{s,t+1} = e^{\tau_0 \xi_0} \begin{bmatrix}
\sigma_{1,t} e^{\tau \xi_1} & \ldots & \sigma_{D,t} e^{\tau \xi_D}
\end{bmatrix}^T \]

Once the sigmas are mutated and/or breeded, the newly generated Individuals are:

\[ I_{r,t+1} = \begin{bmatrix}
w_{0,t} + \sigma_{1,t+1}\xi_1 + \ldots + w_{np,t} + \sigma_{D,t+1}\xi_D
\end{bmatrix}^T \]






<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>





<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
<p>
[1] Baraa M. Abed and Wesam M. Jasim. Hybrid approach for multi-objective optimization path planning with
moving target. Indonesian Journal of Electrical Engineering and Computer Science, 29(1):348–357, 2023.<br>
<p class="line-space">This is on a new line with space below.</p>

[2] V. S. Ajith and K. G. Jolly. Hybrid optimization based multi-objective path planning framework for unmanned
aerial vehicles. Cybernetics and Systems, 54(8):1397–1423, 2023. This study presents a novel Multi-Objective
Path Planning (MOPP) framework for optimizing journey distance and safety of unmanned aerial vehicles
(UAVs) in low-altitude urban environments. The framework utilizes a hybrid Deer Hunter Updated Whale
Optimization (DHUWO) algorithm and showcases improved performance over existing models.
</p>



