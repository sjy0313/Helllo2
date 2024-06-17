package ch07_.inheritance.sec03_parent_constructor.exam03;
// getter /setter 생성 -> 화면 우클릭 -> source -> generate getter & setter
// getter(접근자 메소드) -> 클래스 필드의 값을 반환(읽기전용 속성 제공)
// setter(설정자 메소드) -> 클래스  필드 값 설정(데이터 검증 및 유효성 검사를 수행)
/*
 * Getter와 Setter는 객체 지향 프로그래밍에서 캡슐화를 구현하는 데 중요한 역할을 합니다.
 * 이들은 클래스의 필드에 대한 접근을 제어하고, 데이터의 무결성을 유지하며, 클래스 내부의 구현 세부 사항을 숨깁니다.
 * 이를 통해 더 안전하고 유지보수하기 쉬운 코드를 작성할 수 있습니다.
 */
public class Phone {
    // 필드 선언
    private String model;
    private String color;

    // 기본 생성자
    public Phone() {
        System.out.println("Phone() 기본 생성자 실행");
    }

    // 매개변수가 있는 생성자
    public Phone(String model, String color) {
        this.model = model;
        this.color = color;
        System.out.println("Phone(String model, String color) 생성자 실행");
    }

    // Getter for model
    public String getModel() {
        return model; // this 넣어도 되고 안 넣어도 됨.
    }

    // Setter for model
    public void setModel(String model) {
        this.model = model;
    }

    // Getter for color
    public String getColor() {
        return color;
    }

    // Setter for color
    public void setColor(String color) {
        this.color = color;
    }
}
