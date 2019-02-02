//
//  HGDButton.swift
//  HandGestureDetection
//
//  Created by Vasiliy Dumanov on 2/1/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import UIKit

class HGDButton: UIControl {
    private let _icon: UIImage
    private let _side: CGFloat
    private let _imageSide: CGFloat
    private var _imgView: UIImageView!
    
    override var isEnabled: Bool {
        get {
            return super.isEnabled
        }
        set {
            super.isEnabled = newValue
        }
    }
    
    init(icon: UIImage, side: CGFloat = 60, imageSide: CGFloat = 30) {
        _icon = icon
        _side = side
        _imageSide = imageSide
        super.init(frame: .zero)
        setup()
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    private func setup() {
        setContentCompressionResistancePriority(.required, for: .horizontal)
        setContentCompressionResistancePriority(.required, for: .vertical)
        
        layer.cornerRadius = _side / 2
        layer.masksToBounds = true
        
        _imgView = UIImageView().then {
            $0.translatesAutoresizingMaskIntoConstraints = false
            $0.contentMode = .scaleAspectFit
            $0.image = _icon.withRenderingMode(.alwaysTemplate)
//            $0.tintColor = self.tintColor
        }
        addSubview(_imgView)
        _imgView.snp.makeConstraints { make in
            make.center.equalToSuperview()
            make.width.equalTo(_imageSide)
            make.height.equalTo(_imageSide)
        }
    }
    
    override var intrinsicContentSize: CGSize {
        return CGSize(width: _side, height: _side)
    }
}
