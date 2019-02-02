//
//  PredictionViewController.swift
//  HandGestureDetection
//
//  Created by Vasiliy Dumanov on 1/30/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import UIKit
import SnapKit
import Then

typealias FingerLine = (from: CGPoint, to: CGPoint)

class PredictionViewController: UIViewController {
    private var _model: Hand3DNet!
    
    private var _keypoints: [CGPoint]!
    private var _origImg: UIImage!
    private var _handFgScoremap: UIImage!
    private var _handMask: UIImage!
    
    private var _imgView: UIImageView!
    private var _classLblContainer: UIView!
    private var _classLbl: UILabel!
    private var _keypointsLayer: CALayer!
    private var _viewBtn: HGDButton!
    private var _pointsBtn: HGDButton!
    
    private var _keypointsOn = false
    
    private var _imgViewSize: CGSize {
        let screenSize = UIScreen.main.bounds.size
        return CGSize(width: screenSize.width, height: screenSize.height - 160)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        _model = Hand3DNet()
        setupImageView()
        setupKeypointsLayer()
        setupHeader()
        setupFooter()
        setHandImage(
            UIImage(contentsOfFile: Bundle.main.path(forResource: "IMG_0011-5", ofType: "jpg")!)!
        )
    }
    
    private func setupImageView() {
        _imgView = UIImageView(frame: view.bounds).then {
            $0.contentMode = .scaleAspectFit
            $0.backgroundColor = .black
        }
        view.addSubview(_imgView)
        _imgView.snp.makeConstraints { make in
            make.leading.equalToSuperview()
            make.trailing.equalToSuperview()
            make.top.equalToSuperview().offset(60)
            make.bottom.equalToSuperview().offset(-100)
        }
    }
    
    private func setupKeypointsLayer() {
        _keypointsLayer = CALayer().then {
            $0.frame = CGRect(origin: .zero, size: _imgViewSize)
            $0.delegate = self
            $0.isHidden = true
        }
        _imgView.layer.addSublayer(_keypointsLayer)
    }
    
    private func setupFooter() {
        let stack = UIStackView().then {
            $0.translatesAutoresizingMaskIntoConstraints = false
            $0.axis = .horizontal
            $0.distribution = .equalSpacing
        }
        view.addSubview(stack)
        stack.snp.makeConstraints { make in
            make.bottom.equalToSuperview().offset(-20)
            make.leading.equalToSuperview().offset(10)
            make.trailing.equalToSuperview().offset(-10)
        }
        
        _viewBtn = HGDButton(icon: #imageLiteral(resourceName: "layers")).then {
            $0.translatesAutoresizingMaskIntoConstraints = false
            $0.backgroundColor = .white
            $0.tintColor = .black
            $0.isEnabled = false
        }
        stack.addArrangedSubview(_viewBtn)
        let cameraBtn = HGDButton(icon: #imageLiteral(resourceName: "camera")).then {
            $0.translatesAutoresizingMaskIntoConstraints = false
            $0.backgroundColor = .red
            $0.tintColor = .white
        }
        stack.addArrangedSubview(cameraBtn)
        _pointsBtn = HGDButton(icon: #imageLiteral(resourceName: "graph")).then {
            $0.translatesAutoresizingMaskIntoConstraints = false
            $0.backgroundColor = .white
            $0.tintColor = .black
            $0.isEnabled = false
        }
        stack.addArrangedSubview(_pointsBtn)
        
        cameraBtn.addTarget(self, action: #selector(cameraAction), for: .touchUpInside)
        _pointsBtn.addTarget(self, action: #selector(toggleKeypointsAction), for: .touchUpInside)
        _viewBtn.addTarget(self, action: #selector(changeViewAction), for: .touchUpInside)
    }
    
    private func setupHeader() {
        _classLblContainer = UIView().then {
            $0.translatesAutoresizingMaskIntoConstraints = false
            $0.backgroundColor = UIColor.white.withAlphaComponent(0.8)
            $0.isHidden = true
        }
        view.addSubview(_classLblContainer)
        _classLblContainer.snp.makeConstraints{ make in
            make.leading.equalToSuperview()
            make.trailing.equalToSuperview()
            make.top.equalToSuperview()
            make.height.equalTo(60)
        }
        
        _classLbl = UILabel().then {
            $0.translatesAutoresizingMaskIntoConstraints = false
            $0.textColor = .darkGray
            $0.font = UIFont.boldSystemFont(ofSize: 20)
        }
        _classLblContainer.addSubview(_classLbl)
        _classLbl.snp.makeConstraints { make in
            make.center.equalToSuperview()
        }
    }
    
    @objc private func cameraAction() {
        let picker = UIImagePickerController().then {
            $0.sourceType = .camera
            $0.delegate = self
        }
        present(picker, animated: true)
    }
    
    @objc private func toggleKeypointsAction() {
        _keypointsOn = !_keypointsOn
        if _keypointsOn {
            _pointsBtn.do {
                $0.backgroundColor = UIColor.blue.lighterColor(percent: 0.5)
                $0.tintColor = .white
            }
            _keypointsLayer.isHidden = false
        } else {
            _pointsBtn.do {
                $0.backgroundColor = .white
                $0.tintColor = .black
            }
            _keypointsLayer.isHidden = true
        }
    }
    
    @objc private func changeViewAction() {
        let actionVC = UIAlertController(title: nil, message: nil, preferredStyle: .actionSheet)
        actionVC.addAction(
            UIAlertAction(title: "Original Image", style: .default) { [unowned self] _ in
                self._imgView.image = self._origImg
            }
        )
        actionVC.addAction(
            UIAlertAction(title: "Hand Foreground Scoremap", style: .default) { [unowned self] _ in
                self._imgView.image = self._handFgScoremap
            }
        )
        actionVC.addAction(
            UIAlertAction(title: "Hand Mask", style: .default) { [unowned self] _ in
                self._imgView.image = self._handMask
            }
        )
        actionVC.addAction(
            UIAlertAction(title: "Cancel", style: .cancel)
        )
        present(actionVC, animated: true)
    }
    
    private func setHandImage(_ image: UIImage) {
        DispatchQueue.global(qos: .userInteractive).async { [unowned self] in
            let modelOutput = try! self._model.prediction(image: image)
            self._keypoints = self.convert(keypoints: modelOutput.keypoints,
                                           forImageWithSize: image.size,
                                           toViewportWithSize: self._imgViewSize)
            self._origImg = image
            self._handFgScoremap = self.changeAspectRatio(for: modelOutput.handFgScoremap,
                                                          asInOriginalWithSize: image.size)
            self._handMask = self.changeAspectRatio(for: modelOutput.handMask,
                                                          asInOriginalWithSize: image.size)
            DispatchQueue.main.async {
                self._imgView.image = self._origImg
                self._keypointsLayer.setNeedsDisplay()
                self._pointsBtn.isEnabled = true
                self._viewBtn.isEnabled = true
                self._classLbl.text = modelOutput.gestureLabel.capitalized
                self._classLblContainer.isHidden = false
            }
        }
    }
    
    private func convert(keypoints: [CGPoint], forImageWithSize imageSize: CGSize, toViewportWithSize viewportSize: CGSize) -> [CGPoint] {
        let scale: CGFloat
        if viewportSize.width / viewportSize.height < imageSize.width / imageSize.height {
            scale = viewportSize.width / imageSize.width
        } else {
            scale = viewportSize.height / imageSize.height
        }
        return keypoints.map { pt in
            CGPoint(x: round(pt.x * scale + (viewportSize.width - imageSize.width * scale) / 2),
                    y: round(pt.y * scale + (viewportSize.height - imageSize.height * scale) / 2))
        }
    }
    
    private func getFingerLines(from keypoints: [CGPoint]) -> [[FingerLine]] {
        let pointsPerFinger = 4
        let fingerIdOffsets: [Int] = (0...4).map { id in
            id * pointsPerFinger + 1
        }
        let fingerPts: [[CGPoint]] = fingerIdOffsets.map { offset in
            Array(keypoints[offset..<(offset + pointsPerFinger)])
        }
        let fingerPtsWithOrigin: [[CGPoint]] = fingerPts.map { pts in
            [keypoints[0]] + pts.reversed()
        }
        let fingerLines: [[FingerLine]] = fingerPtsWithOrigin.map { pts in
            (0..<pointsPerFinger).map { fromIdx in
                (pts[fromIdx], pts[fromIdx + 1])
            }
        }
        return fingerLines
    }
    
    private func changeAspectRatio(for image: UIImage, asInOriginalWithSize origSize: CGSize) -> UIImage {
        let scale: CGFloat
        if image.size.width / image.size.height > origSize.width / origSize.height {
            scale = image.size.height / origSize.height
        } else {
            scale = image.size.width / origSize.width
        }
        let cropRect = CGRect(x: round((image.size.width - origSize.width * scale) / 2),
                              y: round((image.size.height - origSize.height * scale) / 2),
                              width: round(origSize.width * scale),
                              height: round(origSize.height * scale))
        let newCGImg = image.cgImage!.cropping(to: cropRect)!
        let newImg = UIImage(cgImage: newCGImg)
        return newImg
    }
    
    override var prefersStatusBarHidden: Bool {
        return true
    }
}

extension PredictionViewController : UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        let image = (info[.originalImage] as! UIImage).withFixedOrientation()
        setHandImage(image)
        picker.dismiss(animated: true)
    }
    
    func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        picker.dismiss(animated: true)
    }
}

extension PredictionViewController : CALayerDelegate {
    func draw(_ layer: CALayer, in ctx: CGContext) {
        UIGraphicsPushContext(ctx)
        defer {
            UIGraphicsPopContext()
        }
        ctx.clear(layer.bounds)
        guard let keypoints = _keypoints else { return }
        let fingerLines = getFingerLines(from: keypoints)
        let fingerColors: [UIColor] = [.red, .green, .blue, .yellow, .brown]
        let darkeningPct: Double = 0.15
        for (lines, color) in zip(fingerLines, fingerColors) {
            for (idx, line) in lines.enumerated() {
                let path = UIBezierPath().then {
                    $0.move(to: line.from)
                    $0.addLine(to: line.to)
                    $0.lineWidth = 4
                    $0.lineJoinStyle = .round
                    $0.lineCapStyle = .round
                }
                color.darkerColor(percent: darkeningPct * Double(idx)).setStroke()
                path.stroke()
            }
        }
    }
}
