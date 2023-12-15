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
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Best-README-Template</h3>

  <p align="center">
    An awesome README template to jumpstart your projects!
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
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
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
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

[![Product Name Screen Shot][product-screenshot]](https://example.com)

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



Use the `BLANK_README.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With


* [![NumPy][NumPy-logo]][NumPy-url]
* [![Matplotlib][Matplotlib-logo]][Matplotlib-url]
* [![NetworkX][NetworkX-logo]][NetworkX-url]
* [![Geomdl][Geomdl-logo]][Geomdl-url]
* [![SciPy][SciPy-logo]][SciPy-url]
* [![DEAP][DEAP-logo]][DEAP-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- Methodology And Pseudoalgorithm -->
## Methodology And Pseudoalgorithm

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

  

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

[1] Baraa M. Abed and Wesam M. Jasim. Hybrid approach for multi-objective optimization path planning with
moving target. Indonesian Journal of Electrical Engineering and Computer Science, 29(1):348–357, 2023.
[2] V. S. Ajith and K. G. Jolly. Hybrid optimization based multi-objective path planning framework for unmanned
aerial vehicles. Cybernetics and Systems, 54(8):1397–1423, 2023. This study presents a novel Multi-Objective
Path Planning (MOPP) framework for optimizing journey distance and safety of unmanned aerial vehicles
(UAVs) in low-altitude urban environments. The framework utilizes a hybrid Deer Hunter Updated Whale
Optimization (DHUWO) algorithm and showcases improved performance over existing models.

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
