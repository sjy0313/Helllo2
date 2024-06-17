package ch07_.inheritance.sec02_phone;

// 상속의 이점 : 클래스의 수정을 최소화 할 수 있음 
// 부모 클래스를 수정하면 모든 자식 클새스에 수정효과
public class Phone {
	//필드 선언
	public String model;
	public String color;
	// https://www.tcpschool.com/java/java_methodConstructor_method - 메소드 생성
	//메소드 선언
	public void bell() {// 반대로 부모가 자식의 필드 사용불가
		System.out.println("벨이 울립니다.");
	}

	public void sendVoice(String message) {
		System.out.println("자기: " + message);
	}
	
	public void receiveVoice(String message) {
		System.out.println("상대방: " + message);
	}

	public void hangUp() {
		System.out.println("전화를 끊습니다.");
	}
}