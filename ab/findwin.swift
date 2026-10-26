import CoreGraphics
import Foundation

let opts: CGWindowListOption = [.optionOnScreenOnly]
guard let list = CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]] else {
    print("no windows"); exit(1)
}
var found = false
for w in list {
    let owner = w[kCGWindowOwnerName as String] as? String ?? ""
    if owner.lowercased().contains("glmark") {
        found = true
        let id = w[kCGWindowNumber as String] as? Int ?? 0
        let b = w[kCGWindowBounds as String] as? [String: Any] ?? [:]
        print("WIN \(id) \(b["X"] ?? 0),\(b["Y"] ?? 0) \(b["Width"] ?? 0)x\(b["Height"] ?? 0)")
    }
}
if !found { print("NONE"); exit(2) }
