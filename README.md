# awesome-robot
A list of robot-related information and resources.


## Best Robotics papers in 2024.

### 1. π0: A Vision-Language-Action Flow Model for General Robot Control [arxiv 2024.10]

**Authors**: Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, Ury Zhilinsky
<details>
<summary><b>Abstract</b></summary>
Robot learning holds tremendous promise to unlock the full potential of flexible, general, and dexterous robot systems, as well as to address some of the deepest questions in artificial intelligence. However, bringing robot learning to the level of generality required for effective real-world systems faces major obstacles in terms of data, generalization, and robustness. In this paper, we discuss how generalist robot policies (i.e., robot foundation models) can address these challenges, and how we can design effective generalist robot policies for complex and highly dexterous tasks. We propose a novel flow matching architecture built on top of a pre-trained vision-language model (VLM) to inherit Internet-scale semantic knowledge. We then discuss how this model can be trained on a large and diverse dataset from multiple dexterous robot platforms, including single-arm robots, dual-arm robots, and mobile manipulators. We evaluate our model in terms of its ability to perform tasks in zero shot after pre-training, follow language instructions from people and from a high-level VLM policy, and its ability to acquire new skills via fine-tuning. Our results cover a wide variety of tasks, such as laundry folding, table cleaning, and assembling boxes.
</details>

[📄 Paper](http://arxiv.org/pdf/2410.24164v3) 

[Website](https://physicalintelligence.company/blog/pi0)

---

### 2. Closed-Loop Open-Vocabulary Mobile Manipulation with GPT-4V [arxiv 2024.04]

**Authors**: Peiyuan Zhi, Zhiyuan Zhang, Muzhi Han, Zeyu Zhang, Zhitian Li, Ziyuan Jiao, Baoxiong Jia, Siyuan Huang
<details>
<summary><b>Abstract</b></summary>
Autonomous robot navigation and manipulation in open environments require reasoning and replanning with closed-loop feedback. We present COME-robot, the first closed-loop framework utilizing the GPT-4V vision-language foundation model for open-ended reasoning and adaptive planning in real-world scenarios. We meticulously construct a library of action primitives for robot exploration, navigation, and manipulation, serving as callable execution modules for GPT-4V in task planning. On top of these modules, GPT-4V serves as the brain that can accomplish multimodal reasoning, generate action policy with code, verify the task progress, and provide feedback for replanning. Such design enables COME-robot to (i) actively perceive the environments, (ii) perform situated reasoning, and (iii) recover from failures. Through comprehensive experiments involving 8 challenging real-world tabletop and manipulation tasks, COME-robot demonstrates a significant improvement in task success rate (~25%) compared to state-of-the-art baseline methods. We further conduct comprehensive analyses to elucidate how COME-robot's design facilitates failure recovery, free-form instruction following, and long-horizon task planning.
</details>

[📄 Paper](http://arxiv.org/pdf/2404.10220v1)

[Website](https://come-robot.github.io/)

---

### 3. ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI [arxiv 2024.10]

**Authors**: Stone Tao, Fanbo Xiang, Arth Shukla, Yuzhe Qin, Xander Hinrichsen, Xiaodi Yuan, Chen Bao, Xinsong Lin, Yulin Liu, Tse-kai Chan, Yuan Gao, Xuanlin Li, Tongzhou Mu, Nan Xiao, Arnav Gurha, Zhiao Huang, Roberto Calandra, Rui Chen, Shan Luo, Hao Su
<details>
<summary><b>Abstract</b></summary>
Simulation has enabled unprecedented compute-scalable approaches to robot learning. However, many existing simulation frameworks typically support a narrow range of scenes/tasks and lack features critical for scaling generalizable robotics and sim2real. We introduce and open source ManiSkill3, the fastest state-visual GPU parallelized robotics simulator with contact-rich physics targeting generalizable manipulation. ManiSkill3 supports GPU parallelization of many aspects including simulation+rendering, heterogeneous simulation, pointclouds/voxels visual input, and more. Simulation with rendering on ManiSkill3 can run 10-1000x faster with 2-3x less GPU memory usage than other platforms, achieving up to 30,000+ FPS in benchmarked environments due to minimal python/pytorch overhead in the system, simulation on the GPU, and the use of the SAPIEN parallel rendering system. Tasks that used to take hours to train can now take minutes. We further provide the most comprehensive range of GPU parallelized environments/tasks spanning 12 distinct domains including but not limited to mobile manipulation for tasks such as drawing, humanoids, and dextrous manipulation in realistic scenes designed by artists or real-world digital twins. In addition, millions of demonstration frames are provided from motion planning, RL, and teleoperation. ManiSkill3 also provides a comprehensive set of baselines that span popular RL and learning-from-demonstrations algorithms.
</details>

[📄 Paper](http://arxiv.org/pdf/2410.00425v1) 

[Website](http://maniskill.ai/)

---

### 4. Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots [arxiv 2024.02]

**Authors**: Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng, Russ Tedrake, Shuran Song
<details>
<summary><b>Abstract</b></summary>
We present Universal Manipulation Interface (UMI) -- a data collection and policy learning framework that allows direct skill transfer from in-the-wild human demonstrations to deployable robot policies. UMI employs hand-held grippers coupled with careful interface design to enable portable, low-cost, and information-rich data collection for challenging bimanual and dynamic manipulation demonstrations. To facilitate deployable policy learning, UMI incorporates a carefully designed policy interface with inference-time latency matching and a relative-trajectory action representation. The resulting learned policies are hardware-agnostic and deployable across multiple robot platforms. Equipped with these features, UMI framework unlocks new robot manipulation capabilities, allowing zero-shot generalizable dynamic, bimanual, precise, and long-horizon behaviors, by only changing the training data for each task. We demonstrate UMI's versatility and efficacy with comprehensive real-world experiments, where policies learned via UMI zero-shot generalize to novel environments and objects when trained on diverse human demonstrations.
</details>

[📄 Paper](http://arxiv.org/pdf/2402.10329v3) 

[Website](https://umi-gripper.github.io)

---

### 5. GOAT: GO to Any Thing [arxiv 2023.11]

**Authors**: Matthew Chang, Theophile Gervet, Mukul Khanna, Sriram Yenamandra, Dhruv Shah, So Yeon Min, Kavit Shah, Chris Paxton, Saurabh Gupta, Dhruv Batra, Roozbeh Mottaghi, Jitendra Malik, Devendra Singh Chaplot
<details>
<summary><b>Abstract</b></summary>
In deployment scenarios such as homes and warehouses, mobile robots are expected to autonomously navigate for extended periods, seamlessly executing tasks articulated in terms that are intuitively understandable by human operators. We present GO To Any Thing (GOAT), a universal navigation system capable of tackling these requirements with three key features: a) Multimodal: it can tackle goals specified via category labels, target images, and language descriptions, b) Lifelong: it benefits from its past experience in the same environment, and c) Platform Agnostic: it can be quickly deployed on robots with different embodiments. GOAT is made possible through a modular system design and a continually augmented instance-aware semantic memory that keeps track of the appearance of objects from different viewpoints in addition to category-level semantics. This enables GOAT to distinguish between different instances of the same category to enable navigation to targets specified by images and language descriptions.
</details>

[📄 Paper](http://arxiv.org/pdf/2311.06430v1)

[Website](https://theophilegervet.github.io/projects/goat/)

---

### 6. Open-TeleVision: Teleoperation with Immersive Active Visual Feedback [arxiv 2024.07]

**Authors**: Xuxin Cheng, Jialong Li, Shiqi Yang, Ge Yang, Xiaolong Wang
<details>
<summary><b>Abstract</b></summary>
Teleoperation serves as a powerful method for collecting on-robot data essential for robot learning from demonstrations. The intuitiveness and ease of use of the teleoperation system are crucial for ensuring high-quality, diverse, and scalable data. To achieve this, we propose an immersive teleoperation system Open-TeleVision that allows operators to actively perceive the robot's surroundings in a stereoscopic manner. Additionally, the system mirrors the operator's arm and hand movements on the robot, creating an immersive experience as if the operator's mind is transmitted to a robot embodiment. We validate the effectiveness of our system by collecting data and training imitation learning policies on four long-horizon, precise tasks (Can Sorting, Can Insertion, Folding, and Unloading) for 2 different humanoid robots and deploy them in the real world.
</details>

[📄 Paper](http://arxiv.org/pdf/2407.01512v2) 

[Website](https://robot-tv.github.io/)

---

### 7. PoliFormer: Scaling On-Policy RL with Transformers Results in Masterful Navigators [arxiv 2024.06]

**Authors**: Kuo-Hao Zeng, Zichen Zhang, Kiana Ehsani, Rose Hendrix, Jordi Salvador, Alvaro Herrasti, Ross Girshick, Aniruddha Kembhavi, Luca Weihs
<details span>
<summary><b>Abstract</b></summary>
We present PoliFormer (Policy Transformer), an RGB-only indoor navigation agent trained end-to-end with reinforcement learning at scale that generalizes to the real-world without adaptation despite being trained purely in simulation. PoliFormer uses a foundational vision transformer encoder with a causal transformer decoder enabling long-term memory and reasoning.
</details>

[📄 Paper](http://arxiv.org/pdf/2406.20083v1)

[Website](https://poliformer.allen.ai/)

---

### 8. NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration [arxiv 2023.10]

**Authors**: Ajay Sridhar, Dhruv Shah, Catherine Glossop, Sergey Levine
<details span>
<summary><b>Abstract</b></summary>
Robotic learning for navigation in unfamiliar environments needs to provide policies for both task-oriented navigation and task-agnostic exploration. In this paper, we describe how we can train a single unified diffusion policy to handle both goal-directed navigation and goal-agnostic exploration, with the latter providing the ability to search novel environments, and the former providing the ability to reach a user-specified goal once it has been located.
</details>

[📄 Paper](http://arxiv.org/pdf/2310.07896v1) 

[Website](https://general-navigation-models.github.io/nomad/)

---

### 9. OpenVLA: An Open-Source Vision-Language-Action Model [arxiv 2024.06]

**Authors**: Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, Chelsea Finn
<details span>
<summary><b>Abstract</b></summary>
Large policies pretrained on a combination of Internet-scale vision-language data and diverse robot demonstrations have the potential to change how we teach robots new skills. OpenVLA is a 7B-parameter open-source vision-language-action model trained on a diverse collection of 970k real-world robot demonstrations. OpenVLA demonstrates strong results for generalist manipulation, outperforming closed models.
</details>

[📄 Paper](http://arxiv.org/pdf/2406.09246v3) 

[Website](https://openvla.github.io/)

---

### 10. Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion [arxiv 2024.01]

**Authors**: Tairan He, Chong Zhang, Wenli Xiao, Guanqi He, Changliu Liu, Guanya Shi
<details span>
<summary><b>Abstract</b></summary>
Legged robots navigating cluttered environments must be jointly agile for efficient task execution and safe to avoid collisions with obstacles or humans. This paper introduces Agile But Safe (ABS), a learning-based control framework that enables agile and collision-free locomotion for quadrupedal robots. ABS involves an agile policy to execute agile motor skills amidst obstacles and a recovery policy to prevent failures, collaboratively achieving high-speed and collision-free navigation.
</details>

[📄 Paper](http://arxiv.org/pdf/2401.17583v3) 

[Website](https://agile-but-safe.github.io/)


---

### 11. HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots [arxiv 2024.10]

**Authors**: Tairan He, Wenli Xiao, Toru Lin, Zhengyi Luo, Zhenjia Xu, Zhenyu Jiang, Jan Kautz, Changliu Liu, Guanya Shi, Xiaolong Wang, Linxi Fan, Yuke Zhu
<details>
<summary><b>Abstract</b></summary>
Humanoid whole-body control requires adapting to diverse tasks such as navigation, loco-manipulation, and tabletop manipulation, each demanding a different mode of control. For example, navigation relies on root velocity tracking, while tabletop manipulation prioritizes upper-body joint angle tracking. Existing approaches typically train individual policies tailored to a specific command space, limiting their transferability across modes. We present the key insight that full-body kinematic motion imitation can serve as a common abstraction for all these tasks and provide general-purpose motor skills for learning multiple modes of whole-body control. Building on this, we propose HOVER (Humanoid Versatile Controller), a multi-mode policy distillation framework that consolidates diverse control modes into a unified policy. HOVER enables seamless transitions between control modes while preserving the distinct advantages of each, offering a robust and scalable solution for humanoid control across a wide range of modes. By eliminating the need for policy retraining for each control mode, our approach improves efficiency and flexibility for future humanoid applications.
</details>

[📄 Paper](http://arxiv.org/pdf/2410.21229v1) 

[Website](https://hover-versatile-humanoid.github.io/)

---

### 12. Real-Time Anomaly Detection and Reactive Planning with Large Language Models [arxiv 2024.07]

**Authors**: Rohan Sinha, Amine Elhafsi, Christopher Agia, Matthew Foutter, Edward Schmerling, Marco Pavone
<details>
<summary><b>Abstract</b></summary>
Foundation models, e.g., large language models (LLMs), trained on internet-scale data possess zero-shot generalization capabilities that make them a promising technology towards detecting and mitigating out-of-distribution failure modes of robotic systems. Our approach involves two-stage reasoning using a fast binary anomaly classifier and a slower fallback selection stage utilizing reasoning capabilities of generative LLMs.
</details>

[📄 Paper](http://arxiv.org/pdf/2407.08735v1) 

[Website](https://sites.google.com/view/aesop-llm)

---

### 13. Mobility VLA: Multimodal Instruction Navigation with Long-Context VLMs and Topological Graphs [arxiv 2024.07]

**Authors**: Hao-Tien Lewis Chiang, Zhuo Xu, Zipeng Fu, Mithun George Jacob, Tingnan Zhang, Tsang-Wei Edward Lee, Wenhao Yu, Connor Schenck, David Rendleman, Dhruv Shah, Fei Xia, Jasmine Hsu, Jonathan Hoech, Pete Florence, Sean Kirmani, Sumeet Singh, Vikas Sindhwani, Carolina Parada, Chelsea Finn, Peng Xu, Sergey Levine, Jie Tan
<details>
<summary><b>Abstract</b></summary>
An elusive goal in navigation research is to build an intelligent agent that can understand multimodal instructions including natural language and image, and perform useful navigation. We present Mobility VLA, a hierarchical Vision-Language-Action (VLA) navigation policy that combines the environment understanding and common-sense reasoning power of long-context VLMs and a robust low-level navigation policy.
</details>

[📄 Paper](http://arxiv.org/pdf/2407.07775v2)

[Website](https://www.youtube.com/watch?v=-Tof__Q8_5s&feature=youtu.be)

---

### 14. HumanPlus: Humanoid Shadowing and Imitation from Humans [arxiv 2024.06]

**Authors**: Zipeng Fu, Qingqing Zhao, Qi Wu, Gordon Wetzstein, Chelsea Finn
<details>
<summary><b>Abstract</b></summary>
One of the key arguments for building robots that have similar form factors to human beings is that we can leverage the massive human data for training. Yet, doing so has remained challenging in practice due to the complexities in humanoid perception and control, lingering physical gaps between humanoids and humans in morphologies and actuation, and lack of a data pipeline for humanoids to learn autonomous skills from egocentric vision. In this paper, we introduce a full-stack system for humanoids to learn motion and autonomous skills from human data. We first train a low-level policy in simulation via reinforcement learning using existing 40-hour human motion datasets. This policy transfers to the real world and allows humanoid robots to follow human body and hand motion in real time using only a RGB camera, i.e. shadowing. Through shadowing, human operators can teleoperate humanoids to collect whole-body data for learning different tasks in the real world. Using the data collected, we then perform supervised behavior cloning to train skill policies using egocentric vision, allowing humanoids to complete different tasks autonomously by imitating human skills.
</details>

[📄 Paper](http://arxiv.org/pdf/2406.10454v1) 

[Website](https://humanoid-ai.github.io/)
