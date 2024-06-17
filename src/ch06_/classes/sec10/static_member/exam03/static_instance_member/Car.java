package ch06_.classes.sec10.static_member.exam03.static_instance_member;
/*
 * static member method : 정적 멤버 메소드
 * - 인스턴스화 되지 않아도 사용할 수 있다
 * - this 를 가지고 있지 않다 
 * - 인스턴스 멤버를 사용할 수 없다
 */
public class Car {
	//인스턴스 필드 선언
	int speed;

	
	//인스턴스 메소드 선언
	void run() {
		System.out.println(speed + "으로 달립니다.");
	}

	static void simulate() {
		//객체 생성
		Car myCar = new Car();
		//인스턴스 멤버 사용
		myCar.speed = 200;
		myCar.run();
	}

	public static void main(String[] args) {
		//정적 메소드 호출
		simulate(); // 클래스 생략 가능, 자신이 속한 멤버의 메소드
		// Car.simulate(); 가능
		
		//객체 생성
		Car myCar = new Car();
		//인스턴스 멤버 사용
		myCar.speed = 60;
		myCar.run();
	}
}