<p align="center">
  <img src="./resources/cherub_rainbow_angry_render.png" alt="Logo" width="200">
</p>

# University of Pisa – Master’s Projects Portfolio
This repository collects and organizes the main projects I completed during my **Master's Degree in Computer Science (Artificial Intelligence track)** at the University of Pisa.

It includes projects, mini-projects, paper reviews, and essays developed across various courses in machine learning, computer vision, reinforcement learning, robotics, data mining, ethical issues in IT, and related AI topics.

All projects were completed individually or in small teams (where noted).

---

## 📚 Courses and Projects

|  Course Name & Project                                                         | Grade         | Date                |
|----------------------------------------------------------------------|----------------|----------------------|
| 643AA - **Artificial Intelligence Fundamentals**                         | 29            | 08/04/2022          |
| *(no associated project folder)*                                     |                |                      |
| 646AA - **Computational Mathematics for Learning and Data Analysis**     | 27            | 01/02/2022          |
| [Study of optimization algorithms: conjugate gradient, BFGS, Arnoldi](./CM) |                |                      |
| 309AA - **Data Mining**                                                  | 30            | 19/01/2021          |
| [Customer behavior analysis, clustering, classification, sequential pattern mining](./DM) |                |                      |
| 649AA - **Human Language Technologies**                                  | 30            | 17/05/2023          |
| [Semantic similarity modeling on US patents with transformers and PET](./HLT) |                |                      |
| 651AA - **Intelligent Systems for Pattern Recognition**                  | 28            | 13/07/2021          |
| [Computer vision midterms: segmentation, LDA, adversarial attacks, CycleGAN](./ISPR) |                |                      |
| 654AA - **Machine Learning**                                             | 29            | 10/02/2021          |
| [Comparative study of regression models on CUP dataset](./ML) |                |                      |
| 655AA - **Mobile and Cyber-Physical Systems**                            | 30            | 22/06/2020          |
| [SmartBin IoT system: waste tracking, gamification, backend/frontend](./MCPS) |                |                      |
| 305AA - **Parallel and Distributed Systems: Paradigms and Models**       | 30            | 21/07/2022          |
| [Parallel video motion detection system with FastFlow, threads](./SPM) |                |                      |
| 387AA - **Robotics**                                                     | 30L           | 11/06/2020          |
| [Survey: Control Strategies for Soft Robotic Manipulators](./Rob) |                |                      |
| 659AA - **Social and Ethical Issues in Information Technology**          | 30L           | 04/08/2020          |
| [Essays on anthropomorphism, machine ethics, punishment](./SEI) |                |                      |
| 755AA - **Computational Health Laboratory**                              | 30            | 03/06/2022          |
| [ML for neurodegenerative disease diagnosis, biomarker extraction](./CHL) |                |                      |
| **Reinforcement Learning**                                    | 30L           | 21/12/2021          |
| [Feudal reinforcement learning paper review and analysis](./RL) |                |                      |
| 658AA - **Smart Applications**                                           | 28            | 26/01/2022          |
| [Cone detection for autonomous driving using stereocamera and LiDAR pipelines](./SA) |                |                      |


## 🎓 Thesis

**Title:** Behavior-Aware Transfer Learning in Reinforcement Learning with π-PACT

**Final Grade:** 110/110 cum laude

This thesis explores behavior-aware transfer learning in reinforcement learning, focusing on a novel framework called **π-PACT** (π-vectors-based Policy Adaptation by Characterization and Transfer).

π-PACT employs **policy supervectors** (π-vectors), compact representations of policy behavior, to characterize and compare skills across different tasks. These π-vectors are obtained by adapting a Universal Background Model (UBM) to the state features observed during policy execution.

The framework dynamically monitors a target policy’s learning process and uses supervector distances to:
- Identify **challenging situations**
- Trigger **knowledge transfer** from relevant source policies

### 🔬 Evaluation

The approach was evaluated in the `highway-env` suite, analyzing:
- The representational properties of the π-vectors  
- The effectiveness of the transfer mechanism

The results highlight the **potential of behavior-aware transfer learning**, while also revealing key challenges and limitations to be addressed in future work.

### 📦 Project Repository

The full thesis project and code are available in a separate repository:  
👉 [π-PACT Project Repository](https://github.com/marcopetix/pi-pact-repo)  <!-- replace with actual link -->
