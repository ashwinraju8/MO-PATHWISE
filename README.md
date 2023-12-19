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
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


The project explores the use of Multi-Objective Optimization (MOO) to perform motion planning. Instead of using traditional optimizers like CHOMP, which maximizes a single objective, path smoothness, while performing obstacle avoidance, 
U = f_smoothness + f_obs, the MOO framework looks at multiple objectives while avoiding obstacles. Single objective optimization gives the single best solution for optimizing the criteria, but MOO gives a set of solutions which are all valid and equal in score. 

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

I = {z, &sigma}

These Initial Populations were generated by Dijkstra's Algorithm in conjunction with Yen's Algorithm, by removing a node for every gnerated individual. 

The individuals are comprised of design variables, controls points and weights, and sigmas (step sizes), each of which correspond to the design variables. So, the dimension of one individual is 1 x 6 x number of controls points. It is the sigmas that are being mutated or (sometimes) breeded at every generation of a new population. 


<p>The Breeding operation averages the step sizes of two randomly selected individuals I_{i,t} and I_{j,t} at generation t:</p>

This operation occurs only if the probability of breeding exceeds a randomly generated probabiity. 

<p>The Mutation operation occurs at every generation of a new population. The mutation of the sigmas of two randomly selected individuals I_{i,t} and I_{j,t} at generation \(t\):</p>

Once the sigmas are mutated and/or breeded, the newly generated Individuals are generated by perturbing the previous individuals by the new sigmas. 


At every Generation of new individuals, the population is maintained at a constant size by filtering, or selecting the individuals based on their fitness scores. Here, we use the NSGA-II algorithm to select the best, or optimal, individuals at each generation. 



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
<p>
[1] Baraa M. Abed and Wesam M. Jasim. Hybrid approach for multi-objective optimization path planning with
moving target. Indonesian Journal of Electrical Engineering and Computer Science, 29(1):348–357, 2023.<br>
<p><br></p>
[2] V. S. Ajith and K. G. Jolly. Hybrid optimization based multi-objective path planning framework for unmanned
aerial vehicles. Cybernetics and Systems, 54(8):1397–1423, 2023. This study presents a novel Multi-Objective
Path Planning (MOPP) framework for optimizing journey distance and safety of unmanned aerial vehicles
(UAVs) in low-altitude urban environments. The framework utilizes a hybrid Deer Hunter Updated Whale
Optimization (DHUWO) algorithm and showcases improved performance over existing models.
</p>



