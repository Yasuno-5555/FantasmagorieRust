use fanta_rust::core::persistence::PersistenceManager;
use fanta_rust::core::{Vec2, ID};
use serde::{Deserialize, Serialize};

#[test]
fn test_save_load_vec2() {
    let pm = PersistenceManager::default();
    let id = ID::from_u64(123);
    let pos = Vec2::new(10.0, 20.0);

    pm.save(id, &pos);

    let loaded: Option<Vec2> = pm.load(id);
    assert_eq!(loaded, Some(pos));
}

#[test]
fn test_save_load_custom_struct() {
    #[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
    struct MyState {
        score: i32,
        name: String,
    }

    let pm = PersistenceManager::default();
    let id = ID::from_u64(55);
    let state = MyState {
        score: 100,
        name: "Player1".to_string(),
    };

    pm.save(id, &state);

    let loaded: Option<MyState> = pm.load(id);
    assert_eq!(loaded, Some(state));
}

#[test]
fn test_serialization_roundtrip() {
    let pm = PersistenceManager::default();
    let id = ID::from_u64(42);
    let val = 999;
    pm.save(id, &val);

    let blob = pm.export_blob().expect("Failed to export");

    // Test import
    let pm2 = PersistenceManager::default();
    pm2.import_blob(&blob);

    let loaded: Option<i32> = pm2.load(id);
    assert_eq!(loaded, Some(val));
}
