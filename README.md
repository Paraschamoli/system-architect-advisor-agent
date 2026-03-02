<p align="center">
  <img src="https://raw.githubusercontent.com/getbindu/create-bindu-agent/refs/heads/main/assets/light.svg" alt="bindu Logo" width="200">
</p>

<h1 align="center">System-Architect-Advisor-Agent</h1>

<p align="center">
  <strong>AI System Architect Advisor is an AI-powered system design assistant that transforms natural language project requirements into structured architecture recommendations, infrastructure planning, security strategies, and compliance considerations. It leverages a multi-model reasoning pipeline to generate schema-driven technical analysis, detailed implementation roadmaps, and well-formatted technical documentation through an interactive chat interface.</strong>
</p>

<p align="center">
  <a href="https://github.com/Paraschamoli/System-Architect-Advisor-Agent/actions/workflows/main.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/Paraschamoli/System-Architect-Advisor-Agent/main.yml?branch=main" alt="Build status">
  </a>
  <a href="https://img.shields.io/github/license/Paraschamoli/System-Architect-Advisor-Agent">
    <img src="https://img.shields.io/github/license/Paraschamoli/System-Architect-Advisor-Agent" alt="License">
  </a>
</p>

---

## 📖 Overview

AI System Architect Advisor is an AI-powered system design assistant that transforms natural language project requirements into structured architecture recommendations, infrastructure planning, security strategies, and compliance considerations. It leverages a multi-model reasoning pipeline to generate schema-driven technical analysis, detailed implementation roadmaps, and well-formatted technical documentation through an interactive chat interface.. Built on the [Bindu Agent Framework](https://github.com/getbindu/bindu) for the Internet of Agents.

**Key Capabilities:**
- 🏗️ **System Architecture Design** - Create scalable, maintainable system architectures
- 📊 **Infrastructure Planning** - Design cloud infrastructure and deployment strategies
- 🔒 **Security Strategy** - Develop comprehensive security frameworks and compliance plans
- 🗺️ **Implementation Roadmaps** - Generate detailed technical implementation plans
- � **Technical Documentation** - Produce structured architecture documents and specifications

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- API keys for OpenRouter and Mem0 (both have free tiers)

### Installation

```bash
# Clone the repository
git clone https://github.com/Paraschamoli/System-Architect-Advisor-Agent.git
cd System-Architect-Advisor-Agent

# Create virtual environment
uv venv --python 3.12.9
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
```

### Configuration

Edit `.env` and add your API keys:

| Key | Get It From | Required |
|-----|-------------|----------|
| `OPENROUTER_API_KEY` | [OpenRouter](https://openrouter.ai/keys) | ✅ Yes |
| `MEM0_API_KEY` | [Mem0 Dashboard](https://app.mem0.ai/dashboard/api-keys) | If you want to use Mem0 tools |

### Run the Agent

```bash
# Start the agent
uv run python -m system_architect_advisor_agent

# Agent will be available at http://localhost:3773
```

### Github Setup

```bash
# Initialize git repository and commit your code
git init -b main
git add .
git commit -m "Initial commit"

# Create repository on GitHub and push (replace with your GitHub username)
gh repo create Paraschamoli/System-Architect-Advisor-Agent --public --source=. --remote=origin --push
```

---

## 💡 Usage

### Example Queries

```bash
# Design e-commerce architecture
"Design a microservices architecture for an e-commerce platform handling 1M users daily, with real-time inventory and payment processing"

# Plan fintech system
"Create a system design for a fintech payment processing system that must handle PCI compliance and process 10K transactions/second"

# Migrate to cloud
"Plan the migration strategy for a legacy monolithic application to a cloud-native microservices architecture"

# Real-time system design
"Design a real-time chat application with video calling, supporting 100K concurrent users with sub-second latency"
```

### Input Formats

**Plain Text:**
```
Design a scalable architecture for a social media platform with 10M users, real-time messaging, and content recommendation engine
```

**JSON:**
```json
{
  "project_type": "e-commerce",
  "scale": "enterprise",
  "users": "1000000",
  "requirements": ["real-time inventory", "payment processing", "recommendations"],
  "constraints": ["PCI compliance", "99.9% uptime", "global deployment"],
  "tech_stack": ["python", "react", "postgresql"]
}
```

### Output Structure

The agent returns structured output with:
- **Architecture Diagrams**: System overview, component interactions, data flow
- **Technical Specifications**: API designs, database schemas, service definitions
- **Infrastructure Plans**: Cloud deployment, networking, scaling strategies
- **Security Framework**: Authentication, authorization, compliance guidelines
- **Implementation Roadmap**: Phased deployment, resource requirements, timelines

---

## 🔌 API Usage

The agent exposes a RESTful API when running. Default endpoint: `http://localhost:3773` 

### Quick Start

For complete API documentation, request/response formats, and examples, visit:

📚 **[Bindu API Reference - Send Message to Agent](https://docs.getbindu.com/api-reference/all-the-tasks/send-message-to-agent)**


### Additional Resources

- 📖 [Full API Documentation](https://docs.getbindu.com/api-reference/all-the-tasks/send-message-to-agent)
- 📦 [Postman Collections](https://github.com/GetBindu/Bindu/tree/main/postman/collections)
- 🔧 [API Reference](https://docs.getbindu.com)

---

## 🎯 Skills

### system_architect_advisor_agent (v1.0.0)

**Primary Capability:**
- Transform natural language requirements into structured system architecture recommendations
- Generate comprehensive infrastructure planning and security strategies
- Create detailed implementation roadmaps and technical documentation

**Features:**
- **Multi-Pattern Architecture**: Microservices, serverless, event-driven, DDD designs
- **Cloud-Agnostic Planning**: AWS, Azure, GCP, and multi-cloud strategies
- **Security-First Design**: Authentication, authorization, encryption, compliance frameworks
- **Scalability Analysis**: Performance modeling and capacity planning
- **Technical Documentation**: Architecture Decision Records (ADRs), API specs, deployment guides

**Best Used For:**
- Startup MVP architecture planning and technology stack selection
- Enterprise system modernization and cloud migration strategies
- Technical due diligence and architecture review processes
- Team training and architectural best practices dissemination

**Not Suitable For:**
- Actual code implementation or deployment execution
- Real-time system monitoring or incident response
- Detailed cost analysis (provides estimates only)
- Proprietary system reverse engineering

**Performance:**
- Average processing time: ~30-45 seconds per architecture design
- Max concurrent requests: 10
- Memory per request: ~512MB
- Supported project scales: Startup to Enterprise level

---

## 🐳 Docker Deployment

### Local Docker Setup

```bash
# Build and run with Docker Compose
docker-compose up --build

# Agent will be available at http://localhost:3773
```

### Docker Configuration

The agent runs on port `3773` and requires:
- `OPENROUTER_API_KEY` environment variable
- `MEM0_API_KEY` environment variable

Configure these in your `.env` file before running.

### Production Deployment

```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d
```

---

## 🌐 Deploy to bindus.directory

Make your agent discoverable worldwide and enable agent-to-agent collaboration.

### Setup GitHub Secrets

```bash
# Authenticate with GitHub
gh auth login

# Set deployment secrets
gh secret set BINDU_API_TOKEN --body "<your-bindu-api-key>"
gh secret set DOCKERHUB_TOKEN --body "<your-dockerhub-token>"
```

Get your keys:
- **Bindu API Key**: [bindus.directory](https://bindus.directory) dashboard
- **Docker Hub Token**: [Docker Hub Security Settings](https://hub.docker.com/settings/security)

### Deploy

```bash
# Push to trigger automatic deployment
git push origin main
```

GitHub Actions will automatically:
1. Build your agent
2. Create Docker container
3. Push to Docker Hub
4. Register on bindus.directory

---

## 🛠️ Development

### Project Structure

```
System-Architect-Advisor-Agent/
├── system_architect_advisor_agent/
│   ├── skills/
│   │   └── system_architect_advisor_agent/
│   │       ├── skill.yaml          # Skill configuration
│   │       └── __init__.py
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py                     # Agent entry point
│   └── agent_config.json           # Agent configuration
├── tests/
│   └── test_main.py
├── .env.example
├── docker-compose.yml
├── Dockerfile.agent
└── pyproject.toml
```

### Running Tests

```bash
make test              # Run all tests
make test-cov          # With coverage report
```

### Code Quality

```bash
make format            # Format code with ruff
make lint              # Run linters
make check             # Format + lint + test
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run manually
uv run pre-commit run -a
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature` 
3. Commit your changes: `git commit -m 'Add amazing feature'` 
4. Push to the branch: `git push origin feature/amazing-feature` 
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Powered by Bindu

Built with the [Bindu Agent Framework](https://github.com/getbindu/bindu)

**Why Bindu?**
- 🌐 **Internet of Agents**: A2A, AP2, X402 protocols for agent collaboration
- ⚡ **Zero-config setup**: From idea to production in minutes
- 🛠️ **Production-ready**: Built-in deployment, monitoring, and scaling

**Build Your Own Agent:**
```bash
uvx cookiecutter https://github.com/getbindu/create-bindu-agent.git
```

---

## 📚 Resources

- 📖 [Full Documentation](https://Paraschamoli.github.io/System-Architect-Advisor-Agent/)
- 💻 [GitHub Repository](https://github.com/Paraschamoli/System-Architect-Advisor-Agent/)
- 🐛 [Report Issues](https://github.com/Paraschamoli/System-Architect-Advisor-Agent/issues)
- 💬 [Join Discord](https://discord.gg/3w5zuYUuwt)
- 🌐 [Agent Directory](https://bindus.directory)
- 📚 [Bindu Documentation](https://docs.getbindu.com)

---

<p align="center">
  <strong>Built with 💛 by the team from Amsterdam 🌷</strong>
</p>

<p align="center">
  <a href="https://github.com/Paraschamoli/System-Architect-Advisor-Agent">⭐ Star this repo</a> •
  <a href="https://discord.gg/3w5zuYUuwt">💬 Join Discord</a> •
  <a href="https://bindus.directory">🌐 Agent Directory</a>
</p>

#   S y s t e m - A r c h i t e c t - A d v i s o r - A g e n t  
 