# A Personalized Dense Retrieval Framework for Unified Information Access (UIA)
Hansi Zeng, Surya Kallumadi, Zaid Alibadi, Rodrigo Nogueira, Hamed Zamani

This repo provides source code for the SIGIR'23 paper: [A Personalized Dense Retrieval Framework for
Unified Information Access](). 

Developing a universal model that can efficiently and effectively respond to a wide range of information access requests—from retrieval to recommendation to question answering—has been a long-lasting goal in the information retrieval community. This paper argues that the flexibility, efficiency, and effectiveness brought by the recent development in dense retrieval and approximate nearest neighbor search have smoothed the path towards achieving this goal. We develop a generic and extensible dense retrieval framework, called UIA, that can handle a wide range of (personalized)
information access requests, such as keyword search, query by example, and complementary item recommendation. Our proposed approach extends the capabilities of dense retrieval models for ad-hoc retrieval tasks by incorporating user-specific preferences through the development of a personalized attentive network. This allows for a more tailored and accurate personalized information access experience. Our experiments on real-world e-commerce data
suggest the feasibility of developing universal information access models by demonstrating significant improvements even compared to competitive baselines specifically developed for each of these individual information access tasks. This work opens up a number of fundamental research directions for future exploration

<p align="center">
  <img align="center" src="https://github.com/HansiZeng/UIA/blob/main/architecture.png" width="850" />
</p>
