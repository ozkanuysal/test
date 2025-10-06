MLOps Engineer
Case Study
Welcome!
Problem Overview
Design and implement a machine learning platform that efficiently manages GPU
resources for a team of data scientists and ML engineers.
The Challenge
Your team needs infrastructure that supports two distinct workflows:
**Interactive Development**: Data scientists experiment with models using small
datasets and minimal GPU resources
**Production Training**: Once experiments are validated, the same code should scale
to large datasets and full GPU clusters
The system must maximize GPU utilization while providing seamless transitions
between development and production environments.
Core Requirements
Build a solution that demonstrates:
**Interactive Environment**: Enable GPU-based model development and testing
**Job Scheduling**: Queue and execute large training jobs based on resource
availability
**Resource Management**: Efficiently allocate GPUs across different workload types
**Scalability**: Same codebase works for both small experiments and large
production runs
**Data Access**: Connect to multiple data sources within your pipeline
Implementation Task
Create a working fine-tuning pipeline using any pre-trained model of your choice.
Your implementation should show both interactive development and production
deployment scenarios.
What We're Looking For
**Technical Implementation (40%)**
Clean, well-structured code
Scalable architecture design
Proper error handling and monitoring
**Problem-Solving Approach (30%)**
How well you address the stated requirements
Creative solutions to resource management challenges
Understanding of distributed systems concepts
**Technology Choices (20%)**
Justified selection of tools and platforms
Clear understanding of trade-offs
Cost-effective resource utilization
**Communication (10%)**
Clear documentation and setup instructions
Ability to explain technical decisions
Deliverables
**1. Working Code**
Complete pipeline implementation
Interactive development environment
Production job scheduling examples
Configuration and deployment scripts
**2. Documentation**
Architecture overview (1-2 pages)
Setup and deployment instructions
Technology selection rationale
Demo scenarios with results
**3. Repository Structure**
```
├── README.md
├── src/ # Source code
├── config/ # Configuration files
├── examples/ # Usage examples
└── docs/ # Additional documentation
```
Submission
Submit a GitHub repository link containing your complete solution. Include a brief cost
analysis if using cloud resources.
Optional: 5-minute demo video of your working system.
Key Questions to Address
How do you prioritize jobs when resources are limited?
What happens when jobs fail or resources become unavailable?
How would your solution scale to 100+ concurrent users?
What monitoring and observability do you include?
Legal Notice
We acknowledge that any work and analysis to be produced during the implementation of
this recruitment task shall be deemed as confidential information and we undertake not to
disclose, share and expose any information or work that obtained from the recruitment task, to
third parties, without your consent. This document and attachments are confidential, legally
privileged and intended solely for the addressee. Access to this document by anyone else is
unauthorized. If you are not the intended recipient, any disclosure, copying, distribution or any
action taken or omitted to be taken in reliance on it, is prohibited and may be unlawful.
Dream, measure, build, repeat
Codeway