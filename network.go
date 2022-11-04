package n2

import (
	"gonum.org/v1/gonum/mat"
	"regexp"
	"strconv"
)

// TODO: Add a simpler Network constructor

var (
	// ATypes can be modified to add custom activation functions when you are parsing Network JSON files
	ATypes = map[string]func() *ActivationLayer{
		"TanH":     NewActTanH,
		"Sigmoid":  NewActSigmoid,
		"ReLU":     NewActReLU,
		"SoftPlus": NewActSoftPlus,
		"SiLU":     NewActSiLU,
	}
	// LayerTypes can be modified to add custom layers types when you are parsing Network JSON files
	// I would like to replace this with something better soon - just not sure how to store layer type int the JSON file#
	LayerTypes = map[string]func() Layer{
		"Dense": func() Layer {
			return new(DenseLayer)
		},
		"Activation": func() Layer {
			return new(ActivationLayer)
		},
		"MaxPool": func() Layer {
			return new(MaxPoolLayer)
		},
		"Corr": func() Layer {
			return new(CorrLayer)
		},
		"Flatten": func() Layer {
			return new(FlattenLayer)
		},
	}
)

// Layer implements Forward, Backward, MarshalJSON and UnmarshalJSON
// * Forward: passes data forward through the network
// * Backward: passes cost gradients backwards, modifying parameters according to the learn rate
type Layer interface {
	Forward([]*mat.Dense) []*mat.Dense
	Backward([]*mat.Dense, float64) []*mat.Dense
	MarshalJSON() ([]byte, error)
	UnmarshalJSON([]byte) error
}

type CostFunc struct {
	cFunc func(output, label []*mat.Dense) float64
	pFunc func(output, label []*mat.Dense) []*mat.Dense
}

// Network is a struct containing a neural network
// To save a network, you can use json.Marshal
// To load a saved network you can use json.Unmarshal
type Network struct {
	Layers    []Layer
	LearnRate float64
}

// NewNetwork is a simple constructor for Network
func NewNetwork(layers []Layer, learnRate float64) (n *Network) {
	n = &Network{
		Layers:    layers,
		LearnRate: learnRate,
	}

	return n
}

// Feedforward passes inputs through the neural network and gives the output.
// Layers will also then cache their activations if required
func (n *Network) Feedforward(inputs []*mat.Dense) []*mat.Dense {
	var activation []*mat.Dense
	copyT3(&activation, &inputs)

	for _, layer := range n.Layers {
		activation = layer.Forward(activation)
	}

	return activation
}

// Backpropagate propagates the cost gradient backwards through the Layers
// The cost gradient is calculated using mean squared error
func (n *Network) Backpropagate(inputs, label []*mat.Dense) (float64, float64) {
	outputs := n.Feedforward(inputs)
	err := mSqErr(outputs, label)

	errGrad := mSqErrPrime(outputs, label)

	for i := len(n.Layers) - 1; i >= 0; i-- {
		errGrad = n.Layers[i].Backward(errGrad, n.LearnRate)
	}

	return err, closeEnough(outputs, label)
}

func (n *Network) UnmarshalJSON(bytes []byte) error {
	rType := regexp.MustCompile("{\"(LType)\"\\s*:\"(\\w+)\"[,\":\\w\\[\\]/+=]+}")
	rLearnRate := regexp.MustCompile("\"(LearnRate)\"\\s*:([\\w.]+)")

	layerMatches := rType.FindAllStringSubmatch(string(bytes), -1)

	n.Layers = make([]Layer, len(layerMatches))

	for i, layerMatch := range layerMatches {
		n.Layers[i] = LayerTypes[layerMatch[2]]()
		err := n.Layers[i].UnmarshalJSON([]byte(layerMatch[0]))
		if err != nil {
			return err
		}
	}

	ls := rLearnRate.FindStringSubmatch(string(bytes))

	var err error
	n.LearnRate, err = strconv.ParseFloat(ls[2], 64)
	if err != nil {
		return err
	}

	return nil
}
