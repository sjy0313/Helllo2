package ch06_.classes.sec06.cars.exam03;
/*
 * 접근 제어자
public: 모든 클래스에서 접근 가능
protected: 동일 패키지 및 하위 클래스에서 접근 가능
default (아무것도 지정하지 않음): 동일 패키지에서 접근 가능
private: 동일 클래스에서만 접근 가능
 */
public class Car {
	//필드 선언
	
	String company = "현대자동차";
	String model = "그랜저";
	String color = "검정";
	
	int maxSpeed = 350;
	int speed;
// 자바에서 클래스 내부에 main 메소드를 정의하는 것은 프로그램의 시작점을 지정하는 것(파이썬에서는 클래스 내부에서 main메소드 정의 불가)
// main 메소드는 JVM(Java Virtual Machine)이 자바 애플리케이션을 실행할 때 호출하는 진입점(entry point)입니다.
	
	// main 메소드는 항상 public, static, void이어야 하며, 매개변수로 String[] args를 받아야 합니다.
	public static void main(String[] args) {
		Car car = new Car(); // 클래스 변수 = new class(); -> class(); 생성자 호출(객체 초기화[=필드 초기화/메소드를 호출해서 객체를 사용할 준비하는 것] 기능)
		// 리턴 타입은 없고 이름은 클래스 이름과 동일
		//Car 객체 생성
		//String company = "우리자동차";
		String company = car.company; // 복사해옴. 
		
		System.out.println("제작회사: " + company); // 제작회사: 현대자동차
		
		System.out.println(company == car.company); // true
		
		company = "우리자동차";
		
		System.out.println(company == car.company); // false
		
		System.out.println("local.company: " + company);  // local.company: 우리자동차
		System.out.println("car.company: " + car.company); // car.company: 현대자동차
		
		
		
		
		
		
	}
}