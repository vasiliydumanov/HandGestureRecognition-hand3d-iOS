//
//  Hand3DNet.swift
//  HandGestureDetection
//
//  Created by Vasiliy Dumanov on 1/26/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import Metal
import MetalPerformanceShaders
import CoreVideo
import CoreML
import CoreImage
import UIKit

struct Hand3DNetOutput {
    let keypoints: [CGPoint]
    let handFgScoremap: UIImage
    let handMask: UIImage
    let gestureLabel: String
}

final class Hand3DNet {
    private let _handSegNet: HandSegNet
    private let _poseNet: PoseNet
    private let _gestureNet: GestureNet
    
    private let _device: MTLDevice
    
    private let _textureRegion: MTLRegion
    private let _bytesPerRow: Int
    private let _texture_0: MTLTexture
    private let _texture_1: MTLTexture
    private let _texture_2: MTLTexture
    
    private let _identityPipeline: MTLComputePipelineState
    private let _softmaxPipeline: MTLComputePipelineState
    private let _replaceMaxMinPipeline: MTLComputePipelineState
    private let _roundPipeline: MTLComputePipelineState
    private let _minMaxKernel: MPSImageStatisticsMinAndMax
    private let _dilateKernel: MPSImageDilate
    private let _multiplyKernel: MPSImageMultiply
    
    private let _threadgroupCount: MTLSize
    private let _threadgroupSize: MTLSize
    
    private let _commandQueue: MTLCommandQueue
    
    init() {
        _handSegNet = HandSegNet()
        _poseNet = PoseNet()
        _gestureNet = GestureNet()
        
        _device = MTLCreateSystemDefaultDevice()!
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.pixelFormat = .r32Float
        textureDescriptor.width = 320
        textureDescriptor.height = 240
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        _texture_0 = _device.makeTexture(descriptor: textureDescriptor)!
        _texture_1 = _device.makeTexture(descriptor: textureDescriptor)!
        _texture_2 = _device.makeTexture(descriptor: textureDescriptor)!
        
        _bytesPerRow = MemoryLayout<Float32>.size * 320
        _textureRegion = MTLRegion(
            origin: MTLOrigin(x: 0, y: 0, z: 0),
            size: MTLSize(width: 320, height: 240, depth: 1))
        
        let library = _device.makeDefaultLibrary()!
        
        let idFunc = library.makeFunction(name: "identityKernel")!
        _identityPipeline = try! _device.makeComputePipelineState(function: idFunc)
        
        let softmaxFunc = library.makeFunction(name: "softmaxKernel")!
        _softmaxPipeline = try! _device.makeComputePipelineState(function: softmaxFunc)
        
        let replaceMaxMinFunc = library.makeFunction(name: "replaceMaxMinKernel")!
        _replaceMaxMinPipeline = try! _device.makeComputePipelineState(function: replaceMaxMinFunc)
        
        let roundFunc = library.makeFunction(name: "roundKernel")!
        _roundPipeline = try! _device.makeComputePipelineState(function: roundFunc)
        
        _minMaxKernel = MPSImageStatisticsMinAndMax(device: _device)
        let dilateFilterArr: [Float] = Array<Float>(repeating: 1.0 / (21 * 21), count: 21 * 21)
        _dilateKernel = MPSImageDilate(device: _device, kernelWidth: 21, kernelHeight: 21, values: dilateFilterArr)
        _multiplyKernel = MPSImageMultiply(device: _device)
        
        _threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        _threadgroupCount = MTLSize(
            width: (320  + _threadgroupSize.width -  1) / _threadgroupSize.width,
            height: (240 + _threadgroupSize.height - 1) / _threadgroupSize.height,
            depth: 1)
        
        _commandQueue = _device.makeCommandQueue()!
    }
    
    func prediction(image: UIImage) throws -> Hand3DNetOutput {
        let totalStartTime = CACurrentMediaTime()
        let resizedImg = resizeImage(image)
        let handSegStartTime = CACurrentMediaTime()
        let handScoremap = try _handSegNet.prediction(image: resizedImg)
        let handScoremapImg = getImage(from: handScoremap.fgScoremap)
        let handSegElapsed = CACurrentMediaTime() - handSegStartTime
        print("HandSegNet - \(handSegElapsed)s")
        let (croppedHandImg, cropCenter, scale, handMask) = cropImageWithHandScoremap(imageBuffer: resizedImg, handScoremap: handScoremap)
        let poseNetStartTime = CACurrentMediaTime()
        let poseNetOutput = try _poseNet.prediction(image: croppedHandImg)
        let poseNetElapsed = CACurrentMediaTime() - poseNetStartTime
        print("PoseNet - \(poseNetElapsed)s")
        let origKeypoints = detectKeypoints(poseNetOutput.keypointScoremaps)
        print("origKeypoints: \(origKeypoints)")
        let gestureNetStartTime = CACurrentMediaTime()
        let gestureLabel = try predictGesture(origKeypoints)
        let gestureNetElapsed = CACurrentMediaTime() - gestureNetStartTime
        print("GestureNet - \(gestureNetElapsed)s")
        let resizedKeypoints = transform(keypoints: origKeypoints, toResizedImageBoundsWithCropCenter: cropCenter, scale: scale)
        let transformedKeypoints = transform(keypoints: resizedKeypoints, toOriginalImageBoundsWithSize: image.size)
        let totalElapsed = CACurrentMediaTime() - totalStartTime
        print("Total - \(totalElapsed)")
        return Hand3DNetOutput(keypoints: transformedKeypoints,
                               handFgScoremap: handScoremapImg,
                               handMask: handMask,
                               gestureLabel: gestureLabel)
    }
    
    private func predictGesture(_ keypoints: [CGPoint]) throws -> String {
        let normalizedKeypoints = try MLMultiArray(shape: [2 * keypoints.count] as [NSNumber], dataType: .float32)
        let normalizedKeypointsPtr = normalizedKeypoints.dataPointer.assumingMemoryBound(to: Float32.self)
        for (idx, keypoint) in keypoints.enumerated() {
            normalizedKeypointsPtr[2 * idx] = Float32(keypoint.y / 256.0)
            normalizedKeypointsPtr[2 * idx + 1] = Float32(keypoint.x / 256.0)
        }
        let preds = try _gestureNet.prediction(keypoints: normalizedKeypoints)
        return preds.classLabel
    }
    
    private func resizeImage(_ img: UIImage) -> CVPixelBuffer {
        let startTime = CACurrentMediaTime()
        let fullRect = CGRect(x: 0, y: 0, width: 320, height: 240)
        
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(fullRect.width), Int(fullRect.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard (status == kCVReturnSuccess) else { fatalError() }
        
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let pixelData = CVPixelBufferGetBaseAddress(pixelBuffer!)
        
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: Int(fullRect.width), height: Int(fullRect.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer!), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        context?.translateBy(x: 0, y: fullRect.height)
        context?.scaleBy(x: 1.0, y: -1.0)
        
        let newImgSize: CGSize
        if img.size.width / img.size.height > fullRect.width / fullRect.height {
            newImgSize = CGSize(
                width: fullRect.width,
                height: (fullRect.width / img.size.width) * img.size.height)
        } else {
            newImgSize = CGSize(
                width: (fullRect.height / img.size.height) * img.size.width,
                height: fullRect.height)
        }
        
        UIGraphicsPushContext(context!)
        UIColor.black.setFill()
        context?.fill(fullRect)
        img.draw(in: CGRect(
            x: (fullRect.width - newImgSize.width) / 2,
            y: (fullRect.height - newImgSize.height) / 2,
            width: newImgSize.width,
            height: newImgSize.height))
        UIGraphicsPopContext()
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let timeElapsed = CACurrentMediaTime() - startTime
        print("resizeImage - \(timeElapsed)s")
        return pixelBuffer!
    }
    
    private func cropImageWithHandScoremap(imageBuffer: CVPixelBuffer, handScoremap: HandSegNetOutput) -> (crop: CVPixelBuffer, center: CGPoint, scale: CGFloat, handMask: UIImage) {
        var startTime = CACurrentMediaTime()
        _texture_0.replace(
            region: _textureRegion,
            mipmapLevel: 0,
            withBytes: handScoremap.bgScoremap.dataPointer,
            bytesPerRow: _bytesPerRow)
        _texture_1.replace(
            region: _textureRegion,
            mipmapLevel: 0,
            withBytes: handScoremap.fgScoremap.dataPointer,
            bytesPerRow: _bytesPerRow)
        
        let commandBuffer = _commandQueue.makeCommandBuffer()!
        
        let softmaxEncoder = commandBuffer.makeComputeCommandEncoder()!
        softmaxEncoder.setComputePipelineState(_softmaxPipeline)
        softmaxEncoder.setTexture(_texture_0, index: 0)
        softmaxEncoder.setTexture(_texture_1, index: 1)
        softmaxEncoder.setTexture(_texture_2, index: 2)
        softmaxEncoder.dispatchThreadgroups(_threadgroupCount, threadsPerThreadgroup: _threadgroupSize)
        softmaxEncoder.endEncoding()

        _minMaxKernel.encode(commandBuffer: commandBuffer, sourceTexture: _texture_2, destinationTexture: _texture_1)

        let replaceMaxMinEncoder = commandBuffer.makeComputeCommandEncoder()!
        replaceMaxMinEncoder.setComputePipelineState(_replaceMaxMinPipeline)
        replaceMaxMinEncoder.setTexture(_texture_0, index: 0)
        replaceMaxMinEncoder.setTexture(_texture_1, index: 1)
        replaceMaxMinEncoder.setTexture(_texture_2, index: 2)
        replaceMaxMinEncoder.dispatchThreadgroups(_threadgroupCount, threadsPerThreadgroup: _threadgroupSize)
        replaceMaxMinEncoder.endEncoding()

        let roundEncoder = commandBuffer.makeComputeCommandEncoder()!
        roundEncoder.setComputePipelineState(_roundPipeline)
        roundEncoder.setTexture(_texture_2, index: 2)
        roundEncoder.dispatchThreadgroups(_threadgroupCount, threadsPerThreadgroup: _threadgroupSize)
        roundEncoder.endEncoding()

        for _ in 0..<32 {
            _dilateKernel.encode(commandBuffer: commandBuffer, sourceTexture: _texture_0, destinationTexture: _texture_1)
            _multiplyKernel.encode(commandBuffer: commandBuffer, primaryTexture: _texture_1, secondaryTexture: _texture_2, destinationTexture: _texture_0)
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let handMaskPtr = UnsafeMutablePointer<Float32>.allocate(capacity: 320 * 240)
        _texture_0.getBytes(
            UnsafeMutableRawPointer(handMaskPtr),
            bytesPerRow: _bytesPerRow,
            from: _textureRegion,
            mipmapLevel: 0)
        
        let handMaskCIImg = CIImage(mtlTexture: _texture_0,
                                    options: [
                                        CIImageOption.colorSpace : CGColorSpaceCreateDeviceGray()
                                    ])!.transformed(by: CGAffineTransform(scaleX: 1, y: -1))
        let handMaskUIImg = handMaskCIImg.toUIImage()
        
        var timeElapsed = CACurrentMediaTime() - startTime
        print("Metal Compute Pipeline - \(timeElapsed)s")
        startTime = CACurrentMediaTime()
        
        var (minX, minY, maxX, maxY) = (320, 240, 0, 0)
        for y in 0..<240 {
            for x in 0..<320 {
                let pixelValue = handMaskPtr[y * 320 + x]
                if pixelValue > 0 {
                    if x < minX {
                        minX = x
                    }
                    if x > maxX {
                        maxX = x
                    }
                    if y < minY {
                        minY = y
                    }
                    if y > maxY {
                        maxY = y
                    }
                }
            }
        }
        handMaskPtr.deallocate()
        
        let cropSize = CGFloat(max(maxX - minX, maxY - minY)) * 1.25
        let scaleCrop = CGFloat(min(max(256 / cropSize, 0.25), 5))
        let cropCenter = CGPoint(x: round(scaleCrop * (CGFloat(minX) + CGFloat(maxX - minX) / 2.0)),
                                 y: round(scaleCrop * (CGFloat(minY) + CGFloat(maxY - minY) / 2.0)))
        let cropCenterFlipped = CGPoint(x: round(scaleCrop * (CGFloat(minX) + CGFloat(maxX - minX) / 2.0)),
                                        y: round(scaleCrop * (CGFloat(240 - minY) + CGFloat(minY - maxY) / 2.0)))
        let cropRect = CGRect(x: cropCenterFlipped.x - 128,
                              y: cropCenterFlipped.y - 128,
                              width: 256,
                              height: 256)
        
        let fullImg = CIImage(cvImageBuffer: imageBuffer)
        let croppedHandImg = fullImg
            .transformed(by: CGAffineTransform(scaleX: scaleCrop, y: scaleCrop))
            .transformed(by: CGAffineTransform(translationX: -cropRect.minX, y: -cropRect.minY))
            .cropped(to: CGRect(origin: .zero, size: cropRect.size))
        
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer : CVPixelBuffer?
        CVPixelBufferCreate(kCFAllocatorDefault, 256, 256, kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        let ciContext = CIContext()

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        ciContext.render(croppedHandImg,
                         to: pixelBuffer!,
                         bounds: CGRect(x: 0, y: 0, width: 256, height: 256),
                         colorSpace: colorSpace)
        CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
        timeElapsed = CACurrentMediaTime() - startTime
        print("Cropping image - \(timeElapsed)s")
        return (crop: pixelBuffer!, center: cropCenter, scale: scaleCrop, handMask: handMaskUIImg)
    }
    
    private func detectKeypoints(_ keypointsScoremap: MLMultiArray) -> [CGPoint] {
        let ksPtr = keypointsScoremap.dataPointer.assumingMemoryBound(to: Float32.self)
        var keypoints: [CGPoint] = []
        for scoremapId in 0..<21 {
            var maxVal = -CGFloat.infinity
            var maxCoords = CGPoint.zero
            for y in 0..<256 {
                for x in 0..<256 {
                    let val = CGFloat(ksPtr[scoremapId * 256 * 256 + y * 256 + x])
                    if val > maxVal {
                        maxVal = val
                        maxCoords = CGPoint(x: x, y: y)
                    }
                }
            }
            keypoints.append(maxCoords)
        }
        return keypoints
    }
    
    private func transform(keypoints: [CGPoint], toResizedImageBoundsWithCropCenter cropCenter: CGPoint, scale: CGFloat) -> [CGPoint] {
        let transorfmedKeypoints = keypoints.map { pt in
            CGPoint(x: (pt.x - 128 + cropCenter.x) / scale,
                    y: (pt.y - 128 + cropCenter.y) / scale)
        }
        return transorfmedKeypoints
    }
    
    private func transform(keypoints: [CGPoint], toOriginalImageBoundsWithSize size: CGSize) -> [CGPoint] {
        let inputSize = CGSize(width: 320, height: 240)
        let scale: CGFloat
        if size.width / size.height > inputSize.width / inputSize.height {
            scale = inputSize.width / size.width
        } else {
            scale = inputSize.height / size.height
        }
        return keypoints.map { pt in
            CGPoint(x: (pt.x - (inputSize.width - size.width * scale) / 2.0) / scale,
                    y: (pt.y - (inputSize.height - size.height * scale) / 2.0) / scale)
        }
    }
    
    private func getImage(from mlArray: MLMultiArray) -> UIImage {
        var minVal: Float = .greatestFiniteMagnitude
        var maxVal: Float = -.greatestFiniteMagnitude
        let dataPtr = mlArray.dataPointer.assumingMemoryBound(to: Float32.self)
        for idx in 0..<mlArray.count {
            let val = dataPtr[idx]
            if val < minVal {
                minVal = val
            }
            if val > maxVal {
                maxVal = val
            }
        }
        let cgImg = mlArray.cgImage(min: Double(minVal), max: Double(maxVal))!
        let uiImg = UIImage(cgImage: cgImg)
        return uiImg
    }
}
