//��ʾ��
#include <vector>
#include <cmath>
#include <map>
namespace nn
{
    class Neuron;
    class Network;
    typedef std::map<Neuron*, double> Links;
    
    //�����
    class ActivationFunction
    {
    public:
        //�������
        virtual double  Output(double x) = 0;
        //��������
        virtual double der(double x)=0;
    protected:

    };

    //ReLU
    class ReLUFunction :public ActivationFunction
    {
    public:
        //�������
        virtual double  Output(double x)
        {
            return x > 0.0 ? x : 0.0;
        }
        //��������
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
        //�������
        virtual double  Output(double x)
        {
            return x ;
        }
        //��������
        virtual double der(double x)
        {
            return 1;
        }
    protected:

    };

    //��ʧ����
    class ComputeLossFunction
    {
    public:
        //������ʧ
        virtual double Error(double output,double target) = 0;
        //��������
        virtual double der(double output, double target)=0;
    protected:

    };
    //������Mean Squared Error��MSE����ʧ
    class MSEFunction :public ComputeLossFunction
    {
    public:
        //������ʧ
        virtual double Error(double output, double target)
        {
            return 0.5 * ::pow((output-target), 2);
        }
        //��������
        virtual double der(double output, double target)
        {
            return (output - target);
        }
    protected:

    };

    //��Ԫ
    class Neuron
    {
    public:
        Neuron(ActivationFunction* funcActivation =NULL, ComputeLossFunction* funcComputeLoss = NULL);
        virtual ~Neuron();
        //ǰ�򴫲�
        virtual bool Forward();
        //���򴫲�
        //virtual bool Back(double target);
        //��ȡ���
        virtual double GetOutput();
        //��ȡ��Ȩ����
        virtual double GetNetOutput();
        //�������� �������Ȩ��
        virtual bool LinkNeuron(Neuron* pInput);
        //�������� �������Ȩ��
        virtual bool LinkNeuron(std::vector<Neuron*>& pInputs);
        //��������
        virtual bool LinkNeuron(Neuron* pInput, double fweight);

        //�������
        virtual bool SetNetOutput(double fOutPut);

        //���²���
        virtual bool UpdateParamters(double fLearnRateing);

        //��ȡĳһ�������Ȩ��
        virtual double GetWeight(Neuron* pInput);
        //����Ȩ��
        virtual void SetWeight(Neuron* pInput,double fWeight);

        //����ƫ��
        virtual void SetBiases(double fBiases);
    public:
        //ƫ��
        double m_fBiases;
        //���
        double m_fOutput;
        //��Ȩ���룬δ�����
        double m_fNetOutput;
        //��¼���
        double m_fdelta;

        //��������
        Links m_Links;

        ActivationFunction* m_funcActivation;
        ComputeLossFunction* m_funcComputeLoss;
    };

    //�����
    typedef  std::vector<Neuron*>  NetworkLayer;

    //������
    class Network
    {
    public:
        Network(int InputSize, double fLearnRating=0.0000001);
        virtual ~Network();

        //�ں�������һ������
        bool AddLayer(int NeuronSize);
        //��ĳһ������һ�������Ľڵ�
        bool AddLayerNode(int nIndex, int NeuronSize = 1);
        size_t GetLayerSize();
        size_t GetLayerNodeSize(int nIndex);
        Neuron* GetNeuron(int nLayerIndex, int nNeuronIndex);
        //ǰ�򴫲�
        virtual bool Forward(std::vector<double> InputData, std::vector<double>& OutputData);
        //���򴫲�,����Ȩ��
        virtual bool Back(std::vector<double>& Target);

        virtual double GetLossError(std::vector<double>& Target);
        
    protected:
        std::vector< NetworkLayer > m_vecLayer;

        //ѧϰ��
        double m_fLearnRating;

        ActivationFunction* m_funcActivation;
        ComputeLossFunction* m_funcComputeLoss;
    };
}