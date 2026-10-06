import NNC
import XCTest

final class OptimizerTests: XCTestCase {

  private func runGradScalerStep(gradient: Float32, kind: DeviceKind = .CPU) -> (
    parameter: Float32, scale: Float
  ) {
    let dynamicGraph = DynamicGraph()
    let parameter: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(kind, .NC(1, 2))
    parameter.full(1)
    let gradientTensor: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(kind, .NC(1, 2))
    gradientTensor.full(gradient)
    var optimizer = SGDOptimizer(
      dynamicGraph, nesterov: false, rate: 0.1, scale: 1, decay: 0, momentum: 0,
      dampening: 0)
    optimizer.parameters = [parameter]
    let loss = parameter .* gradientTensor
    loss.backward(to: parameter)
    var scaler = GradScaler(scale: 8, growthFactor: 2, backoffFactor: 0.5, growthInterval: 2_000)
    scaler.step(&optimizer)
    return (parameter.toCPU().rawValue[0, 0], scaler.scale)
  }

  func testGradScalerRejectsNonFiniteGradientsOnMPS() throws {
    for gradient in [Float32.nan, Float32.infinity, -Float32.infinity] {
      let result = runGradScalerStep(gradient: gradient, kind: .GPU(0))
      XCTAssertEqual(result.parameter, 1)
      XCTAssertEqual(result.scale, 4)
    }
  }

  func testGradScalerSkippedAdamStepDoesNotCorruptState() throws {
    let dynamicGraph = DynamicGraph()
    let parameter = dynamicGraph.variable(Tensor<Float32>([1], .CPU, .C(1)))
    var optimizer = AdamOptimizer(dynamicGraph, rate: 0.1)
    optimizer.parameters = [parameter]
    var scaler = GradScaler(scale: 8, growthFactor: 2, backoffFactor: 0.5, growthInterval: 2_000)

    var coefficient = dynamicGraph.constant(Tensor<Float32>([Float32.infinity], .CPU, .C(1)))
    (parameter * coefficient).backward(to: parameter)
    scaler.step(&optimizer)
    XCTAssertEqual(parameter[0], 1)
    XCTAssertEqual(optimizer.step, 1)

    coefficient = dynamicGraph.constant(Tensor<Float32>([4], .CPU, .C(1)))
    (parameter * coefficient).backward(to: parameter)
    scaler.step(&optimizer)
    XCTAssertTrue(parameter[0].isFinite)
    XCTAssertNotEqual(parameter[0], 1)
    XCTAssertEqual(optimizer.step, 2)
  }

  func testGradScalerRejectsNonFiniteModelGradients() throws {
    for gradient in [Float32.nan, Float32.infinity, -Float32.infinity] {
      let dynamicGraph = DynamicGraph()
      let model = Dense(count: 1)
      let input = dynamicGraph.variable(Tensor<Float32>([1], .CPU, .C(1)))
      var optimizer = SGDOptimizer(
        dynamicGraph, nesterov: false, rate: 0.1, scale: 1, decay: 0, momentum: 0,
        dampening: 0)
      optimizer.parameters = [model.parameters]
      let output = model(input)
      let before = output[0]
      let coefficient = dynamicGraph.constant(Tensor<Float32>([gradient], .CPU, .C(1)))
      (output * coefficient).backward(to: input)
      var scaler = GradScaler(
        scale: 8, growthFactor: 2, backoffFactor: 0.5, growthInterval: 2_000)
      scaler.step(&optimizer)
      XCTAssertEqual(model(input)[0], before)
      XCTAssertEqual(scaler.scale, 4)
    }
  }

  func testGradScalerRejectsNonFiniteGradients() throws {
    let finite = runGradScalerStep(gradient: 8)
    XCTAssertEqual(finite.parameter, 0.9, accuracy: 1e-6)
    XCTAssertEqual(finite.scale, 8)

    for gradient in [Float32.nan, Float32.infinity, -Float32.infinity] {
      let result = runGradScalerStep(gradient: gradient)
      XCTAssertEqual(result.parameter, 1)
      XCTAssertEqual(result.scale, 4)
    }
  }

  func testSGDOnModel() throws {
    let dynamicGraph = DynamicGraph()
    let linear = Dense(count: 1)
    let z = dynamicGraph.variable(Tensor<Float32>([5], .CPU, .C(1)))
    var sgd = SGDOptimizer(
      dynamicGraph, nesterov: false, rate: 0.01, scale: 1, decay: 0.01, momentum: 0, dampening: 0)
    sgd.parameters = [linear.parameters]
    for i in 0..<100 {
      let x: DynamicGraph.Tensor<Float32>
      if i % 2 == 1 {
        x = dynamicGraph.variable(.CPU, .NC(2, 1))
        x[0, 0] = 10
        x[1, 0] = 10
      } else {
        x = dynamicGraph.variable(.CPU, .C(1))
        x[0] = 10
      }
      var y = Functional.log(x)
      y = linear(y)
      y = y - z
      let f = y .* y
      f.backward(to: x)
      sgd.step()
    }
    let x: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .C(1))
    x[0] = 10
    var y = Functional.log(x)
    y = linear(y)
    XCTAssertEqual(y[0], 5, accuracy: 1e-2)
  }

  func testSGDOnGraph() throws {
    let dynamicGraph = DynamicGraph()
    let weight: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .C(1))
    let bias = dynamicGraph.variable(Tensor<Float32>([0], .CPU, .C(1)))
    weight.rand(-1...1)
    let z = dynamicGraph.variable(Tensor<Float32>([5], .CPU, .C(1)))
    var sgd = SGDOptimizer(
      dynamicGraph, nesterov: false, rate: 0.01, scale: 1, decay: 0.01, momentum: 0, dampening: 0)
    sgd.parameters = [weight, bias]
    for i in 0..<100 {
      let x: DynamicGraph.Tensor<Float32>
      if i % 2 == 1 {
        x = dynamicGraph.variable(.CPU, .NC(2, 1))
        x[0, 0] = 10
        x[1, 0] = 10
      } else {
        x = dynamicGraph.variable(.CPU, .C(1))
        x[0] = 10
      }
      var y = Functional.log(x)
      y = y * weight + bias
      y = y - z
      let f = y .* y
      f.backward(to: x)
      sgd.step()
    }
    let x: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .C(1))
    x[0] = 10
    var y = Functional.log(x)
    y = y * weight + bias
    XCTAssertEqual(y[0], 5, accuracy: 1e-2)
  }

  func testCopyTrainedModel() throws {
    let dynamicGraph = DynamicGraph()
    let linear = Dense(count: 1)
    let z = dynamicGraph.variable(Tensor<Float32>([5], .CPU, .C(1)))
    var sgd = SGDOptimizer(
      dynamicGraph, nesterov: false, rate: 0.01, scale: 1, decay: 0.01, momentum: 0, dampening: 0)
    sgd.parameters = [linear.parameters]
    for i in 0..<100 {
      let x: DynamicGraph.Tensor<Float32>
      if i % 2 == 1 {
        x = dynamicGraph.variable(.CPU, .NC(2, 1))
        x[0, 0] = 10
        x[1, 0] = 10
      } else {
        x = dynamicGraph.variable(.CPU, .C(1))
        x[0] = 10
      }
      var y = Functional.log(x)
      y = linear(y)
      y = y - z
      let f = y .* y
      f.backward(to: x)
      sgd.step()
    }
    let x: DynamicGraph.Tensor<Float32> = dynamicGraph.variable(.CPU, .C(1))
    x[0] = 10
    var y = Functional.log(x)
    y = linear(y)
    XCTAssertEqual(y[0], 5, accuracy: 1e-2)
    let copy = linear.copied()
    copy.parameters.copy(from: linear.parameters)
    var yCopy = Functional.log(x)
    yCopy = copy(yCopy)
    XCTAssertEqual(yCopy[0], 5, accuracy: 1e-2)
  }

  static let allTests = [
    ("testGradScalerRejectsNonFiniteGradients", testGradScalerRejectsNonFiniteGradients),
    ("testGradScalerRejectsNonFiniteGradientsOnMPS", testGradScalerRejectsNonFiniteGradientsOnMPS),
    ("testGradScalerSkippedAdamStepDoesNotCorruptState", testGradScalerSkippedAdamStepDoesNotCorruptState),
    ("testGradScalerRejectsNonFiniteModelGradients", testGradScalerRejectsNonFiniteModelGradients),
    ("testSGDOnModel", testSGDOnModel),
    ("testSGDOnGraph", testSGDOnGraph),
    ("testCopyTrainedModel", testCopyTrainedModel),
  ]
}
