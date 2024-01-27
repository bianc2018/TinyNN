#include "TinyNN.h"

static nn::ReLUFunction gst_ActivationFunction;
static nn::MSEFunction gst_ComputeLossFunction;

nn::Network::Network(int InputSize, double fLearnRating)
    :m_fLearnRating(fLearnRating)
{
    m_funcActivation = &gst_ActivationFunction;
    m_funcComputeLoss = &gst_ComputeLossFunction;

    AddLayer(InputSize);
}

nn::Network::~Network()
{
    for (auto &L : m_vecLayer)
    {
        for (auto &n : L)
        {
            delete n;
        }
    }
    m_vecLayer.clear();
}

bool nn::Network::AddLayer(int NeuronSize)
{
    NetworkLayer NewLayer;
    size_t nLayerSize = GetLayerSize();
    for (int i = 0; i < NeuronSize; ++i)
    {
        Neuron* pNeuron = new  Neuron(m_funcActivation, m_funcComputeLoss);
        NewLayer.push_back(pNeuron);
        if (nLayerSize != 0)
        {
            pNeuron->LinkNeuron(m_vecLayer[nLayerSize - 1]);
        }
    }
    m_vecLayer.push_back(NewLayer);
   
    return false;
}

bool nn::Network::AddLayerNode(int nIndex, int NeuronSize)
{
    if(nIndex<0|| nIndex>= GetLayerSize())
        return false;

    for (int i = 0; i < NeuronSize; ++i)
    {
        Neuron* pNeuron = new  Neuron(m_funcActivation, m_funcComputeLoss);
        m_vecLayer[nIndex].push_back(pNeuron);
        //连接前面的层
        if (nIndex - 1 >= 0)
        {
            pNeuron->LinkNeuron(m_vecLayer[nIndex - 1]);
        }
        //链接到后面的层
        if (nIndex + 1 < GetLayerSize())
        {
            for (auto& LN : m_vecLayer[nIndex + 1])
            {
                LN->LinkNeuron(pNeuron);
           }
        }
    }
    return true;
}

size_t nn::Network::GetLayerSize()
{
    return m_vecLayer.size();
}

size_t nn::Network::GetLayerNodeSize(int nIndex)
{
    if (nIndex < 0 || nIndex >= GetLayerSize())
        return 0;
    return m_vecLayer[nIndex].size();
}

nn::Neuron* nn::Network::GetNeuron(int nLayerIndex, int nNeuronIndex)
{
    if (nLayerIndex < 0 || nLayerIndex >= GetLayerSize())
        return nullptr;

    if (nNeuronIndex < 0 || nNeuronIndex >= GetLayerNodeSize(nLayerIndex))
        return nullptr;

    return m_vecLayer[nLayerIndex][nNeuronIndex];
}

bool nn::Network::Forward(std::vector<double> InputData, std::vector<double>& OutputData)
{
    if (GetLayerSize() <= 0|| InputData.size()!= m_vecLayer[0].size())
        return false;

    //更新到输入层
    for (int i = 0; i < InputData.size(); ++i)
        m_vecLayer[0][i]->SetNetOutput(InputData[i]);

    //第二层开始推算
    for (int i = 1; i < GetLayerSize(); ++i)
        for (int j = 0; j < m_vecLayer[i].size(); ++j)
            m_vecLayer[i][j]->Forward();

    //结果
    if (GetLayerSize() >= 1)
    {
        NetworkLayer& OutputLayer = m_vecLayer[GetLayerSize() - 1];
        for (int i = 0; i < OutputLayer.size(); ++i)
            OutputData.push_back(OutputLayer[i]->GetOutput());
    }
    return true;
}

bool nn::Network::Back(std::vector<double>& Target)
{
    if (GetLayerSize() <= 0 || Target.size() != m_vecLayer[GetLayerSize()-1].size())
        return false;

    //计算输出层误差（Compute Output Layer Error）
    NetworkLayer& OutputLayer = m_vecLayer[GetLayerSize() - 1];
    for (int i = 0; i < OutputLayer.size(); ++i)
    {
        OutputLayer[i]->m_fdelta = m_funcComputeLoss->der(OutputLayer[i]->m_fOutput, Target[i])/**m_funcActivation->der(OutputLayer[i]->m_fNetOutput)*/;
    }
    
    //反向传播误差（Backpropagate Error）
    for (int i = m_vecLayer.size() - 2; i > 0; --i)
    {
        NetworkLayer& NowLayer = m_vecLayer[i];
        NetworkLayer& NextLayer = m_vecLayer[i+1];

        for (auto& now : NowLayer)
        {
            //该神经元连接到输出层神经元的权重和对应的输出层神经元的误差项之和
            double ftemp = 0.0;
            for (auto& next : NextLayer)
            {
                ftemp+= next->GetWeight(now) * next->m_fdelta;
            }

            now->m_fdelta = m_funcActivation->der(now->m_fNetOutput) * ftemp;
        }
    }

    //计算梯度和更新参数
    for (int i = 1; i < m_vecLayer.size(); ++i)
    {
        NetworkLayer& NowLayer = m_vecLayer[i];
        for (auto& node: NowLayer)
        {
            node->UpdateParamters(m_fLearnRating);
        }
    }
    return false;
}

double nn::Network::GetLossError(std::vector<double>& Target)
{
    if (GetLayerSize() <= 0 || Target.size() != m_vecLayer[GetLayerSize() - 1].size())
        return 0.0;

    double fLossError = 0.0;
    NetworkLayer& OutputLayer = m_vecLayer[GetLayerSize() - 1];
    for (int i = 0; i < OutputLayer.size(); ++i)
    {
        fLossError+= m_funcComputeLoss->Error(OutputLayer[i]->m_fOutput, Target[i]) ;
    }
    return fLossError;
}

nn::Neuron::Neuron(ActivationFunction* funcActivation, ComputeLossFunction* funcComputeLoss)
    :m_fBiases(0.1)
    , m_fOutput(0.0)
    , m_fNetOutput(0.0)
    , m_fdelta(0.0)
    , m_funcActivation(funcActivation)
    , m_funcComputeLoss(funcComputeLoss)
{
    if (!m_funcActivation)
        m_funcActivation = &gst_ActivationFunction;
    if (!m_funcComputeLoss)
        m_funcComputeLoss = &gst_ComputeLossFunction;
}

nn::Neuron::~Neuron()
{
}

bool nn::Neuron::Forward()
{
    //加权输入
    m_fNetOutput = m_fBiases;
    for (auto it : m_Links)
    {
        m_fNetOutput += it.first->GetOutput() * it.second;
    }
    m_fOutput = gst_ActivationFunction.Output(m_fNetOutput);
    return true;
}

double nn::Neuron::GetOutput()
{
    return m_fOutput;
}

double nn::Neuron::GetNetOutput()
{
    return m_fNetOutput;
}

bool nn::Neuron::LinkNeuron(Neuron* pInput)
{
    const int weight_base = 100000000;
    double fweight = rand() % weight_base;
    fweight /= weight_base;
    //fweight = 1;
    return LinkNeuron(pInput, fweight);
}

bool nn::Neuron::LinkNeuron(std::vector<Neuron*>& pInputs)
{
    for (auto& pInput : pInputs)
    {
        LinkNeuron(pInput);
    }
    return true;
}

bool nn::Neuron::LinkNeuron(Neuron* pInput, double fweight)
{
    //已存在 覆盖
    m_Links[pInput] = fweight;
    return true;
}

bool nn::Neuron::SetNetOutput(double fOutPut)
{
    m_fNetOutput = fOutPut;
    m_fOutput = gst_ActivationFunction.Output(fOutPut);
    return true;
}

bool nn::Neuron::UpdateParamters(double fLearnRateing)
{
    //更新偏置
    printf("Update m_fBiases:%lf->%lf\n", m_fBiases, m_fBiases - fLearnRateing * m_fdelta);
    m_fBiases -= fLearnRateing * m_fdelta;
    
    //更新输入权重
    for (auto &it : m_Links)
    {
        printf("Update weight:%lf->%lf\n", it.second, it.second - fLearnRateing * (it.first->GetOutput() * m_fdelta));
        it.second -= fLearnRateing * (it.first->GetOutput() * m_fdelta);
    }
    return true;
}

double nn::Neuron::GetWeight(Neuron* pInput)
{
    Links::iterator it = m_Links.find(pInput);
    if(m_Links.end() == it)
        return 0.0;
    return it->second;
}

void nn::Neuron::SetWeight(Neuron* pInput, double fWeight)
{
    Links::iterator it = m_Links.find(pInput);
    if (m_Links.end() == it)
        return ;
    it->second = fWeight;
}

void nn::Neuron::SetBiases(double fBiases)
{
    m_fBiases = fBiases;
}
