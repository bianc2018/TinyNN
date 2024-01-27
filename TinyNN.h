//演示用
#include <vector>
#include <cmath>
#include <map>
namespace nn
{
    class Neuron;
    class Network;
    typedef std::map<Neuron*, double> Links;
    
    //激活函数
    class ActivationFunction
    {
    public:
        //计算输出
        virtual double  Output(double x) = 0;
        //导数计算
        virtual double der(double x)=0;
    protected:

    };

    //ReLU
    class ReLUFunction :public ActivationFunction
    {
    public:
        //计算输出
        virtual double  Output(double x)
        {
            return x > 0.0 ? x : 0.0;
        }
        //导数计算
        virtual double der(double x) 
        {
            return x > 0.0 ? 1 : 0.0;
        }
    protected:

    };

    //NULL
    class NULLFunction :public ActivationFunction
    {
    public:
        //计算输出
        virtual double  Output(double x)
        {
            return x ;
        }
        //导数计算
        virtual double der(double x)
        {
            return 1;
        }
    protected:

    };

    //损失函数
    class ComputeLossFunction
    {
    public:
        //计算损失
        virtual double Error(double output,double target) = 0;
        //导数计算
        virtual double der(double output, double target)=0;
    protected:

    };
    //均方误差（Mean Squared Error，MSE）损失
    class MSEFunction :public ComputeLossFunction
    {
    public:
        //计算损失
        virtual double Error(double output, double target)
        {
            return 0.5 * ::pow((output-target), 2);
        }
        //导数计算
        virtual double der(double output, double target)
        {
            return (output - target);
        }
    protected:

    };

    //神经元
    class Neuron
    {
    public:
        Neuron(ActivationFunction* funcActivation =NULL, ComputeLossFunction* funcComputeLoss = NULL);
        virtual ~Neuron();
        //前向传播
        virtual bool Forward();
        //反向传播
        //virtual bool Back(double target);
        //获取输出
        virtual double GetOutput();
        //获取加权输入
        virtual double GetNetOutput();
        //链接输入 生成随机权重
        virtual bool LinkNeuron(Neuron* pInput);
        //链接输入 生成随机权重
        virtual bool LinkNeuron(std::vector<Neuron*>& pInputs);
        //链接输入
        virtual bool LinkNeuron(Neuron* pInput, double fweight);

        //设置输出
        virtual bool SetNetOutput(double fOutPut);

        //更新参数
        virtual bool UpdateParamters(double fLearnRateing);

        //获取某一个输入的权重
        virtual double GetWeight(Neuron* pInput);
        //设置权重
        virtual void SetWeight(Neuron* pInput,double fWeight);

        //设置偏置
        virtual void SetBiases(double fBiases);
    public:
        //偏置
        double m_fBiases;
        //输出
        double m_fOutput;
        //加权输入，未激活的
        double m_fNetOutput;
        //记录误差
        double m_fdelta;

        //输入链接
        Links m_Links;

        ActivationFunction* m_funcActivation;
        ComputeLossFunction* m_funcComputeLoss;
    };

    //网络层
    typedef  std::vector<Neuron*>  NetworkLayer;

    //神经网络
    class Network
    {
    public:
        Network(int InputSize, double fLearnRating=0.0000001);
        virtual ~Network();

        //在后面新增一层网络
        bool AddLayer(int NeuronSize);
        //在某一层新增一定数量的节点
        bool AddLayerNode(int nIndex, int NeuronSize = 1);
        size_t GetLayerSize();
        size_t GetLayerNodeSize(int nIndex);
        Neuron* GetNeuron(int nLayerIndex, int nNeuronIndex);
        //前向传播
        virtual bool Forward(std::vector<double> InputData, std::vector<double>& OutputData);
        //反向传播,更新权重
        virtual bool Back(std::vector<double>& Target);

        virtual double GetLossError(std::vector<double>& Target);
        
    protected:
        std::vector< NetworkLayer > m_vecLayer;

        //学习率
        double m_fLearnRating;

        ActivationFunction* m_funcActivation;
        ComputeLossFunction* m_funcComputeLoss;
    };
}