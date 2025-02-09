<!-- markdownlint-disable MD033 -->

# 🤖  AI Agentic Lab: Build Single and Multi-Agent Systems in Azure

Welcome to the **Agentic System Design Lab**! This is your go-to space for exploring, designing, and implementing agent-based AI systems in Azure. Our mission is to help you understand and build single-agent systems, foundational orchestration frameworks, and advanced multi-agent strategies. We leverage the power of [**Azure Foundry (Azure AI Agents Service)**](https://azure.microsoft.com/en-us/products/ai-foundry/?msockid=0b24a995eaca6e7d3c1dbc1beb7e6fa8#Use-cases-and-Capabilities) and leading AI frameworks like [**Autogen**](https://microsoft.github.io/autogen/0.2/docs/Getting-Started/) and [**Semantic Kernel**](https://learn.microsoft.com/en-us/semantic-kernel/overview/) to provide you with the best resources and tools for building enterprise agentic designs patterns. Our focus is on hands-on learning—empowering you to learn the fundamentals trougth the labs and explore the art of the possible through real-world use cases.

**Let’s build the future of AI, one agent at a time! 🚀**

## What's new ✨

➕ [**Building MaS with Azure AI Agents & Semantic Kernel Agent Framework (Experimental)**](labs/03-building-multi-agent-systems/sk-and-azure-ai-agents.ipynb) lab – Orchestrate single-agent  on Azure AI Agent Services using the experimental Semantic Kernel Agent Framework in Python.  
➕ [**Building MaS with Azure AI Agents & Autogen v0.4 (New Autogen Architecture)**](labs/03-building-multi-agent-systems/autogen-and-azure-ai-agents.ipynb) lab – Orchestrate single-agent on Azure AI Agent Services with the event-driven Autogen v0.4 architecture.

## Contents

1. [🤖 Building Agentic Systems in Azure](#-building-agentic-systems-in-azure)
1. [🧪 Labs](#-labs)
1. [🚀 Uses Cases](#-getting-started)
1. [📚 More Resources](#-other-resources)

## 🤖 Building Agentic Systems in Azure

In today's fast-evolving Agentic AI landscape, staying ahead means embracing rapid experimentation. Our approach in ths repo is to keep it simple and to the point, starting with the development of robust, scalable **enterprise single agents** using the Azure AI Agent Service. These production-ready agents come equipped with integrated tools, persistent memory, traceability, and isolated execution—providing a solid foundation before scaling up.

Then, of course, we'll tackle communication patterns between single agents. Just as clear conversation drives human collaboration, real-time event exchange between agents unlocks their full potential as a cohesive system. By leveraging frameworks like **AutoGen** and **Semantic Kernel**—or even crafting your own— you can establish an event-driven architecture that seamlessly ties everything together (single-agents) to build multi-agent systems.

```text
Multi-Agent Architecture = Σ (Production-Ready Single Agents [tools, memory, traceability, isolation]) + Preferred Framework (e.g., Semantic Kernel, AutoGen)
```

**Breaking It Down**

- **Step 1:** Build robust, scalable single agents with the **Azure AI Agent Service**, managing them as micro-services.
- **Step 2:** For complex tasks, deploy a fleet of specialized agents that communicate seamlessly via an event-driven framework of your choice.

## 🧪 Labs
Ready to dive into developing agentic AI systems? Explore our labs to build, refine, and experiment with cutting-edge agent architectures on Azure.

+ 🧪 **Building Single Agents with Azure AI Agent Service**:  
   - [🧾 Notebook - Building Single Agents with Azure AI Agent Service](labs/01-azure-ai-agents/single-agent-with-azure-ai-agents.ipynb): Master the fundamentals of single-agent architectures and learn how to build managed agent as micro-services.
- 🧪 **Deep Dive into Frameworks**: Discover Semantic Kernel and AutoGen
   - [🧾 Notebook - Intro to Semantic Kernel](labs/02-frameworks/intro-to-semantic-kernel.ipynb)  
   + [🧾 Notebook - Intro to Autogen](labs/02-frameworks/intro-to-autogen-v2.ipynb)
   - [🧾 Notebook - Upgrade to new Autogen Architecture](labs/02-frameworks/upgrade-to-autogen-new-architecture.ipynb): In January 2025, AutoGen released its new generation v0.4—a significant evolution from version 0.2. For a smooth transition, review Notebook 03 before moving on to Notebook 03b to see why this new design is the event we needed.
+ 🧪 **Building Multi-Agent Architectures**:
   - [🧾 Notebook - Intro to Semantick Kernel Agentic Framework](labs/03-building-multi-agent-systems/sk-agent-framework.ipynb): *Caution: The SK Agentic framework is currently in experimental phase.*
   + [🧾 Notebook - Building MaS with Azure AI Agents & Autogen v0.4 (New Autogen Architecture)](labs/03-building-multi-agent-systems/autogen-and-azure-ai-agents.ipynb)
   - [🧾 Notebook - Building MaS with Azure AI Agents & Semantic Kernel Agent Framework (Experimental)](labs/03-building-multi-agent-systems/sk-and-azure-ai-agents.ipynb)

For more details, please visit the Labs [README](labs/README.md).

## 🚀 Uses Cases

**Cooming very soon**

## 📚 More Resources

- **[Azure AI Foundry](https://azure.microsoft.com/en-us/products/ai-foundry/?msockid=0b24a995eaca6e7d3c1dbc1beb7e6fa8#Use-cases-and-Capabilities)**: Develop and deploy custom AI apps and APIs responsibly with a comprehensive platform.
- **[Azure AI Agent Service](https://learn.microsoft.com/en-us/azure/ai-services/agents/overview)**: Learn about Azure AI Agent Service and its capabilities.
- **[AutoGen Documentation](https://microsoft.github.io/autogen/0.2/docs/Getting-Started/)**: Comprehensive guides and API references for AutoGen.
- **[Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/overview/)**: Detailed documentation on Semantic Kernel's features and capabilities.

### Disclaimer

> [!IMPORTANT]
> This software is provided for demonstration purposes only. It is not intended to be relied upon for any purpose. The creators of this software make no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability or availability with respect to the software or the information, products, services, or related graphics contained in the software for any purpose. Any reliance you place on such information is therefore strictly at your own risk.