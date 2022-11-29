package n2

import (
	"bytes"
	"encoding/gob"
	"github.com/e74000/alg"
	"gonum.org/v1/gonum/mat"
)

// TODO: Add a simpler Network constructor

var (
	// LayerTypes
	layerTypes = []Layer{
		&ActivationLayer{},
		&FlattenLayer{},
		&DenseLayer{},
		&MaxPoolLayer{},
		&CorrLayer{},
	}
)

func RegisterLayerType(layer Layer) {
	layerTypes = append(layerTypes, layer)
}

func registerAll() {
	for _, layerType := range layerTypes {
		gob.Register(layerType)
	}

	for _, term := range alg.Terms {
		gob.Register(term)
	}
}

// Layer implements Forward, Backward, MarshalJSON and UnmarshalJSON
// * Forward: passes data forward through the network
// * Backward: passes cost gradients backwards, modifying parameters according to the learn rate
type Layer interface {
	Forward([]*mat.Dense) []*mat.Dense
	Backward([]*mat.Dense, float64) []*mat.Dense
}

// Network is a struct containing a neural network
// To save a network, you can use json.Marshal
// To load a saved network you can use json.Unmarshal
type Network struct {
	Layers    []Layer
	LearnRate float64
}

// NewNetwork is a simple constructor for Network
func NewNetwork(layers ...Layer) (n *Network) {
	n = &Network{
		Layers:    layers,
		LearnRate: 0,
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
func (n *Network) Backpropagate(inputs, label []*mat.Dense) (cost float64, grad []*mat.Dense) {
	outputs := n.Feedforward(inputs)
	err := mSqErr(outputs, label)

	errGrad := mSqErrPrime(outputs, label)

	for i := len(n.Layers) - 1; i >= 0; i-- {
		errGrad = n.Layers[i].Backward(errGrad, n.LearnRate)
	}

	return err, errGrad
}

// ToGob converts the network into a file using the gob format
func (n *Network) ToGob() ([]byte, error) {
	var buffer bytes.Buffer

	registerAll()

	encoder := gob.NewEncoder(&buffer)
	err := encoder.Encode(n)
	if err != nil {
		panic(err)
	}

	return buffer.Bytes(), nil
}

// FromGob converts a file into a network
func (n *Network) FromGob(b []byte) error {
	*n = Network{}

	registerAll()

	decoder := gob.NewDecoder(bytes.NewReader(b))
	err := decoder.Decode(&n)

	if err != nil {
		panic(err)
	}

	return nil
}
