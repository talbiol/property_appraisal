import pytest

from property_appraisal.web import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_index_lists_bundled_zip_codes(client):
    response = client.get("/")
    assert response.status_code == 200
    for zip_code in (b"90001", b"90002", b"90003", b"90005"):
        assert b'value="' + zip_code + b'"' in response.data


@pytest.mark.parametrize("zip_code", ["abc", "99999"])
def test_rejects_unavailable_zip_code(client, zip_code):
    response = client.post("/", data={"zip_code": zip_code})
    assert response.status_code == 400
    assert b"Choose one of the available zip codes" in response.data


def test_appraises_bundled_zip_code(client):
    response = client.post("/", data={"zip_code": "90001"})
    assert response.status_code == 200
    assert b"undervalued listing" in response.data
