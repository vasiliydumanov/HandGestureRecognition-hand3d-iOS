//
//  CIImageExtensions.swift
//  HandGestureDetection
//
//  Created by Vasiliy Dumanov on 1/31/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import Foundation
import CoreImage
import UIKit

extension CIImage {
    func toUIImage() -> UIImage {
        let context = CIContext(options: nil)
        let cgImage = context.createCGImage(self, from: self.extent)!
        let image = UIImage(cgImage: cgImage)
        return image
    }
}
