use fanta_rust::core::{ColorF, FrameArena, Vec2, ID};
use fanta_rust::view::header::ViewType;
use fanta_rust::widgets::UIContext;

#[test]
fn test_plot_builder() {
    let arena = FrameArena::new();
    let mut ui = UIContext::new(&arena);

    let points = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0)];
    let points_ref = ui.arena.alloc_slice(&points);

    ui.plot()
        .title("Test Plot")
        .height(300.0)
        .x_range(0.0, 10.0)
        .y_range(-1.0, 1.0)
        .line(points_ref, ColorF::RED)
        .build();

    let root = ui.root().expect("Root view should exist");
    assert_eq!(root.view_type, ViewType::Plot);
    assert_eq!(root.height.get(), 300.0);

    // Verify PlotData
    let plot_data = root.plot_data.get().expect("PlotData should be set");
    assert_eq!(plot_data.title, Some("Test Plot"));
    assert_eq!(plot_data.x_min, 0.0);
    assert_eq!(plot_data.x_max, 10.0);

    // Verify Items
    assert_eq!(plot_data.items.len(), 1);
    match &plot_data.items[0] {
        fanta_rust::view::plot::PlotItem::Line { points, color, .. } => {
            assert_eq!(points.len(), 2);
            assert_eq!(color, &ColorF::RED);
        }
        _ => panic!("Expected Line item"),
    }
}
