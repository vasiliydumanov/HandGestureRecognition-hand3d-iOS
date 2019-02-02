//
//  AppDelegate.swift
//  HandGestureDetection
//
//  Created by Vasiliy Dumanov on 1/26/19.
//  Copyright © 2019 Distillery. All rights reserved.
//

import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        window = UIWindow(frame: UIScreen.main.bounds)
        window?.rootViewController = PredictionViewController(nibName: nil, bundle: nil)
        window?.makeKeyAndVisible()
        return true
    }

}

