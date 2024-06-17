package ch07_.inheritance.sec02_phone;
// inheritance : 객체 지향 프로그램에서 부모클래스의 필드와 메소드를
// 자식 클래스에게 물려줄 수 있다. 코드를 줄여 개발 시간 단축에 용이
// extends = 상속 [다중상속을 지원하지 않음 -> 부작용이 많음]

public class SmartPhone extends Phone {
	//필드 선언
	public boolean wifi;

	//생성자 선언
	public SmartPhone(String model, String color) {
		this.model = model; // 상속받았기 떄문에 자신의 것처럼 사용(this)
		this.color = color;
	}

	//메소드 선언
	public void setWifi(boolean wifi) {
		this.wifi = wifi;
		System.out.println("와이파이 상태를 변경했습니다.");
	}

	public void internet() {
		System.out.println("인터넷에 연결합니다.");
	}
}